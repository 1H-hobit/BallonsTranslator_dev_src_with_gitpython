from qtpy.QtCore import Signal, Qt, QPointF, QSize, QSizeF, QLineF, QRectF
from qtpy.QtWidgets import QGridLayout, QPushButton, QComboBox, QSizePolicy, QBoxLayout, QCheckBox, QHBoxLayout, QGraphicsView, QStackedWidget, QVBoxLayout, QLabel, QGraphicsPixmapItem, QGraphicsEllipseItem
from qtpy.QtGui import QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFontMetrics

from typing import Union, Tuple, List
import numpy as np
import cv2

from utils.imgproc_utils import enlarge_window
from utils.textblock_mask import canny_flood, connected_canny_flood
from utils.logger import logger
from utils.config import pcfg
from .funcmaps import get_maskseg_method
from .module_manager import ModuleManager
from .image_edit import ImageEditMode, PenShape, PixmapItem, StrokeImgItem
from .configpanel import InpaintConfigPanel
from .custom_widget import Widget, SeparatorWidget, PaintQSlider, ColorPickerLabel
from .canvas import Canvas
from .misc import ndarray2pixmap
from utils.config import DrawPanelConfig, pcfg
from utils.shared import CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.logger import logger as LOGGER
from .drawing_commands import InpaintUndoCommand, StrokeItemUndoCommand

# 定义常量
INPAINT_BRUSH_COLOR = QColor(127, 0, 127, 127)  # 修复工具的默认颜色
MAX_PEN_SIZE = 1000  # 最大画笔尺寸
MIN_PEN_SIZE = 1     # 最小画笔尺寸
TOOLNAME_POINT_SIZE = 13  # 工具名称标签的字体大小

class DrawToolCheckBox(QCheckBox):
    """自定义复选框，用于工具选择"""
    checked = Signal()
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateChanged.connect(self.on_state_changed)

    def mousePressEvent(self, event) -> None:
        """防止已选中的工具被取消选中"""
        if self.isChecked():
            return
        return super().mousePressEvent(event)

    def on_state_changed(self, state: int) -> None:
        """状态变化时发射信号"""
        if self.isChecked():
            self.checked.emit()

class ToolNameLabel(QLabel):
    """自定义工具名称标签，支持固定宽度和字体调整"""
    def __init__(self, fix_width=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setPointSizeF(TOOLNAME_POINT_SIZE)
        fmt = QFontMetrics(font)
        
        # 如果设置了固定宽度，则调整字体大小以适应
        if fix_width is not None:
            self.setFixedWidth(fix_width)
            text_width = fmt.width(self.text())
            if text_width > fix_width * 0.95:
                font_size = TOOLNAME_POINT_SIZE * fix_width * 0.95 / text_width
                font.setPointSizeF(font_size)
        self.setFont(font)

class InpaintPanel(Widget):
    """修复工具配置面板"""
    thicknessChanged = Signal(int)

    def __init__(self, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 厚度滑块
        self.thicknessSlider = PaintQSlider()
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSlider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 厚度布局
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        # 形状选择
        shape_label = ToolNameLabel(100, self.tr('Shape'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle'),     # 圆形
            self.tr('Rectangle'),  # 矩形
            # self.tr('Triangle')  # 三角形（未实现）
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        # 修复器布局
        self.inpaint_layout = inpaint_layout = QHBoxLayout()
        inpaint_layout.addWidget(ToolNameLabel(100, self.tr('Inpainter')))
        self.inpainter_panel = inpainter_panel

        # 主布局
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(inpaint_layout)
        layout.addLayout(thickness_layout)
        layout.addLayout(shape_layout)
        layout.setSpacing(14)

    def on_thickness_changed(self):
        """厚度变化时发射信号"""
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    def showEvent(self, e) -> None:
        """显示时添加修复器选择框"""
        self.inpaint_layout.addWidget(self.inpainter_panel.module_combobox)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        """隐藏时移除修复器选择框"""
        self.inpaint_layout.removeWidget(self.inpainter_panel.module_combobox)
        return super().hideEvent(e)

    @property
    def shape(self):
        """获取当前选择的形状"""
        return self.shapeCombobox.currentIndex()

class PenConfigPanel(Widget):
    """画笔工具配置面板"""
    thicknessChanged = Signal(int)
    colorChanged = Signal(list)
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 厚度滑块
        self.thicknessSlider = PaintQSlider()
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSlider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 透明度滑块
        self.alphaSlider = PaintQSlider()
        self.alphaSlider.setRange(0, 255)
        self.alphaSlider.setValue(255)
        self.alphaSlider.valueChanged.connect(self.on_alpha_changed)

        # 颜色选择器
        self.colorPicker = ColorPickerLabel()
        self.colorPicker.colorChanged.connect(self.on_color_changed)
        
        # 颜色和透明度布局
        color_label = ToolNameLabel(None, self.tr('Color'))
        alpha_label = ToolNameLabel(None, self.tr('Alpha'))
        color_layout = QHBoxLayout()
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.colorPicker)
        color_layout.addWidget(alpha_label)
        color_layout.addWidget(self.alphaSlider)
        
        # 厚度布局
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        # 形状选择
        shape_label = ToolNameLabel(100, self.tr('Shape'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle'),
            self.tr('Rectangle'),
            # self.tr('Triangle')
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        # 主布局
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(color_layout)
        layout.addLayout(thickness_layout)
        layout.addLayout(shape_layout)
        layout.setSpacing(20)

    def on_thickness_changed(self):
        """厚度变化时发射信号"""
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    def on_alpha_changed(self):
        """透明度变化时更新颜色并发射信号"""
        color = self.colorPicker.rgba()
        color = [color[0], color[1], color[2], self.alphaSlider.value()]
        self.colorPicker.setPickerColor(color)
        self.colorChanged.emit(color)

    def on_color_changed(self):
        """颜色变化时发射信号"""
        color = self.colorPicker.rgba()
        color = [color[0], color[1], color[2], self.alphaSlider.value()]
        self.colorChanged.emit(color)

    @property
    def shape(self):
        """获取当前选择的形状"""
        return self.shapeCombobox.currentIndex()

class RectPanel(Widget):
    """矩形工具配置面板"""
    dilate_ksize_changed = Signal()
    method_changed = Signal(int)
    delete_btn_clicked = Signal()
    inpaint_btn_clicked = Signal()
    
    def __init__(self, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 膨胀参数
        self.dilate_label = ToolNameLabel(100, self.tr('Dilate'))
        self.dilate_slider = PaintQSlider()
        self.dilate_slider.setRange(0, 100)
        self.dilate_slider.valueChanged.connect(self.dilate_ksize_changed)
        
        # 方法选择
        self.methodComboBox = QComboBox()
        self.methodComboBox.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.methodComboBox.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.methodComboBox.addItems([
            self.tr('method 1'), 
            self.tr('method 2'),
            self.tr('Use Existing Mask')
        ])
        self.methodComboBox.activated.connect(self.on_inpaint_seg_method_changed)
        
        # 自动修复复选框
        self.autoChecker = QCheckBox(self.tr("Auto"))
        self.autoChecker.setToolTip(self.tr("run inpainting automatically."))
        self.autoChecker.stateChanged.connect(self.on_auto_changed)
        
        # 修复和删除按钮
        self.inpaint_btn = QPushButton(self.tr("Inpaint"))
        self.inpaint_btn.setToolTip(self.tr("Space"))
        self.inpaint_btn.clicked.connect(self.inpaint_btn_clicked)
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setToolTip(self.tr('Ctrl+D'))
        self.delete_btn.clicked.connect(self.delete_btn_clicked)
        self.btnlayout = QHBoxLayout()
        self.btnlayout.addWidget(self.inpaint_btn)
        self.btnlayout.addWidget(self.delete_btn)

        # 修复器布局
        self.inpaint_layout = inpaint_layout = QHBoxLayout()
        inpaint_layout.addWidget(ToolNameLabel(100, self.tr('Inpainter')))
        self.inpainter_panel = inpainter_panel

        # 网格布局
        glayout = QGridLayout()
        glayout.addWidget(self.dilate_label, 0, 0)
        glayout.addWidget(self.dilate_slider, 0, 1)
        glayout.addWidget(self.autoChecker, 1, 0)
        glayout.addWidget(self.methodComboBox, 1, 1)

        # 主布局
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(inpaint_layout)
        layout.addLayout(glayout)
        layout.addLayout(self.btnlayout)
        layout.setSpacing(14)

    def showEvent(self, e) -> None:
        """显示时添加修复器选择框"""
        self.inpaint_layout.addWidget(self.inpainter_panel.module_combobox)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        """隐藏时移除修复器选择框"""
        self.inpaint_layout.removeWidget(self.inpainter_panel.module_combobox)
        return super().hideEvent(e)
        
    def on_inpaint_seg_method_changed(self):
        """修复方法变化时更新配置"""
        pcfg.drawpanel.rectool_method = self.methodComboBox.currentIndex()

    def on_auto_changed(self):
        """自动修复复选框状态变化处理"""
        if self.autoChecker.isChecked():
            self.inpaint_btn.hide()
            self.delete_btn.hide()
            pcfg.drawpanel.rectool_auto = True
        else:
            pcfg.drawpanel.rectool_auto = False
            self.inpaint_btn.show()
            self.delete_btn.show()

    def auto(self) -> bool:
        """是否启用自动修复"""
        return self.autoChecker.isChecked()

    def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """对遮罩进行膨胀处理"""
        if mask is None:
            return None
        ksize = self.dilate_slider.value()
        if ksize == 0:
            return mask
        # 创建椭圆形的结构元素
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
        return cv2.dilate(mask, element)  # 膨胀操作

class DrawingPanel(Widget):
    """主绘图面板，集成所有工具和画布交互"""

    scale_tool_pos: QPointF = None  # 缩放工具的位置

    def __init__(self, canvas: Canvas, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 初始化成员变量
        self.module_manager: ModuleManager = None
        self.canvas = canvas
        self.inpaint_stroke: StrokeImgItem = None  # 当前修复笔划
        self.rect_inpaint_dict: dict = None  # 矩形修复数据
        self.inpaint_mask_array: np.ndarray = None  # 修复遮罩数组
        self.extracted_imask_array: np.ndarray = None  # 提取的遮罩数组

        # 创建遮罩项和缩放圆
        border_pen = QPen(INPAINT_BRUSH_COLOR, 3, Qt.PenStyle.DashLine)
        self.inpaint_mask_item: PixmapItem = PixmapItem(border_pen)
        self.scale_circle = QGraphicsEllipseItem()
        
        # 连接画布信号
        canvas.finish_painting.connect(self.on_finish_painting)
        canvas.finish_erasing.connect(self.on_finish_erasing)
        canvas.ctrl_relesed.connect(self.on_canvasctrl_released)
        canvas.begin_scale_tool.connect(self.on_begin_scale_tool)
        canvas.scale_tool.connect(self.on_scale_tool)
        canvas.end_scale_tool.connect(self.on_end_scale_tool)
        canvas.scalefactor_changed.connect(self.on_canvas_scalefactor_changed)
        canvas.end_create_rect.connect(self.on_end_create_rect)

        # 初始化工具
        self.currentTool: DrawToolCheckBox = None
        self.handTool = DrawToolCheckBox()  # 手形工具
        self.handTool.setObjectName("DrawHandTool")
        self.handTool.checked.connect(self.on_use_handtool)
        self.handTool.stateChanged.connect(self.on_handchecker_changed)
        
        self.inpaintTool = DrawToolCheckBox()  # 修复工具
        self.inpaintTool.setObjectName("DrawInpaintTool")
        self.inpaintTool.checked.connect(self.on_use_inpainttool)
        self.inpaintConfigPanel = InpaintPanel(inpainter_panel)
        self.inpaintConfigPanel.thicknessChanged.connect(self.setInpaintToolWidth)
        self.inpaintConfigPanel.shapeChanged.connect(self.setInpaintShape)
        
        self.rectTool = DrawToolCheckBox()  # 矩形工具
        self.rectTool.setObjectName("DrawRectTool")
        self.rectTool.checked.connect(self.on_use_recttool)
        self.rectTool.stateChanged.connect(self.on_rectchecker_changed)
        self.rectPanel = RectPanel(inpainter_panel)
        self.rectPanel.inpaint_btn_clicked.connect(self.on_rect_inpaintbtn_clicked)
        self.rectPanel.delete_btn_clicked.connect(self.on_rect_deletebtn_clicked)
        self.rectPanel.dilate_ksize_changed.connect(self.on_rectool_ksize_changed)
        
        self.penTool = DrawToolCheckBox()  # 画笔工具
        self.penTool.setObjectName("DrawPenTool")
        self.penTool.checked.connect(self.on_use_pentool)
        self.penConfigPanel = PenConfigPanel()
        self.penConfigPanel.thicknessChanged.connect(self.setPenToolWidth)
        self.penConfigPanel.colorChanged.connect(self.setPenToolColor)
        self.penConfigPanel.shapeChanged.connect(self.setPenShape)

        # 工具布局
        toolboxlayout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        toolboxlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        toolboxlayout.addWidget(self.handTool)
        toolboxlayout.addWidget(self.inpaintTool)
        toolboxlayout.addWidget(self.penTool)
        toolboxlayout.addWidget(self.rectTool)

        # 初始化画笔
        self.canvas.painting_pen = self.pentool_pen = \
            QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.canvas.erasing_pen = self.erasing_pen = QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.inpaint_pen = QPen(INPAINT_BRUSH_COLOR, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        
        # 工具配置堆叠窗口
        self.toolConfigStackwidget = QStackedWidget()
        self.toolConfigStackwidget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        self.toolConfigStackwidget.addWidget(self.inpaintConfigPanel)
        self.toolConfigStackwidget.addWidget(self.penConfigPanel)
        self.toolConfigStackwidget.addWidget(self.rectPanel)

        # 遮罩透明度滑块
        self.maskTransperancySlider = PaintQSlider()
        self.maskTransperancySlider.valueChanged.connect(self.canvas.setMaskTransparencyBySlider)
        masklayout = QHBoxLayout()
        masklayout.addWidget(ToolNameLabel(130, self.tr('Mask Opacity')))
        masklayout.addWidget(self.maskTransperancySlider)

        # 主布局
        layout = QVBoxLayout(self)
        layout.addLayout(toolboxlayout)
        layout.addWidget(SeparatorWidget())
        layout.addWidget(self.toolConfigStackwidget)
        layout.addWidget(SeparatorWidget())
        layout.addLayout(masklayout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def setCurrentToolByName(self, tool_name: str):
        """通过名称设置当前工具"""
        try:
            set_method = f'on_use_{tool_name}tool'
            set_method = getattr(self, set_method)
            set_method()
            if self.currentTool is not None:
                self.currentTool.setChecked(True)
        except:
            LOGGER.error(f'{set_method} not found in drawing panel')

    def shortcutSetCurrentToolByName(self, tool_name: str):
        """通过快捷键设置当前工具"""
        if self.isVisible():
            self.setCurrentToolByName(tool_name)

    def setShortcutTip(self, tool_name: str, shortcut: str):
        """设置工具快捷键提示"""
        try:
            tool = f'{tool_name}Tool'
            tool: QStackedWidget = getattr(self, tool)
            tool.setToolTip(f'{shortcut}')
        except:
            LOGGER.error(f'{tool} not found in drawing panel')

    def initDLModule(self, module_manager: ModuleManager):
        """初始化深度学习模块"""
        self.module_manager = module_manager
        module_manager.canvas_inpaint_finished.connect(self.on_inpaint_finished)
        module_manager.inpaint_thread.inpaint_failed.connect(self.on_inpaint_failed)

    def setInpaintToolWidth(self, width):
        """设置修复工具宽度"""
        self.inpaint_pen.setWidthF(width)
        pcfg.drawpanel.inpainter_width = width
        if self.isVisible():
            self.setInpaintCursor()

    def setInpaintShape(self, shape: int):
        """设置修复工具形状"""
        self.setInpaintCursor()
        pcfg.drawpanel.inpainter_shape = shape
        self.canvas.painting_shape = shape

    def setPenToolWidth(self, width):
        """设置画笔工具宽度"""
        self.pentool_pen.setWidthF(width)
        self.erasing_pen.setWidthF(width)
        pcfg.drawpanel.pentool_width = self.pentool_pen.widthF()
        if self.isVisible():
            self.setPenCursor()

    def setPenToolColor(self, color: Union[QColor, Tuple, List]):
        """设置画笔颜色"""
        if not isinstance(color, QColor):
            color = QColor(*color)
        self.pentool_pen.setColor(color)
        pcfg.drawpanel.pentool_color = [color.red(), color.green(), color.blue(), color.alpha()]
        if self.isVisible():
            self.setPenCursor()
        self.penConfigPanel.colorPicker.setPickerColor(color)
        self.penConfigPanel.alphaSlider.setValue(color.alpha())

    def setPenShape(self, shape: int):
        """设置画笔形状"""
        self.setPenCursor()
        self.canvas.painting_shape = shape
        pcfg.drawpanel.pentool_shape = shape

    def on_use_handtool(self) -> None:
        """切换到手形工具"""
        if self.currentTool is not None and self.currentTool != self.handTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.handTool
        pcfg.drawpanel.current_tool = ImageEditMode.HandTool
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # 设置拖拽模式
        self.canvas.image_edit_mode = ImageEditMode.HandTool

    def on_use_inpainttool(self) -> None:
        """切换到修复工具"""
        if self.currentTool is not None and self.currentTool != self.inpaintTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.inpaintTool
        pcfg.drawpanel.current_tool = ImageEditMode.InpaintTool
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        self.canvas.painting_pen = self.inpaint_pen
        self.canvas.erasing_pen = self.inpaint_pen
        self.canvas.painting_shape = self.inpaintConfigPanel.shape
        self.toolConfigStackwidget.setCurrentWidget(self.inpaintConfigPanel)  # 显示修复配置面板
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setInpaintCursor()

    def on_use_pentool(self) -> None:
        """切换到画笔工具"""
        if self.currentTool is not None and self.currentTool != self.penTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.penTool
        pcfg.drawpanel.current_tool = ImageEditMode.PenTool
        self.canvas.painting_pen = self.pentool_pen
        self.canvas.painting_shape = self.penConfigPanel.shape
        self.canvas.erasing_pen = self.erasing_pen
        self.canvas.image_edit_mode = ImageEditMode.PenTool
        self.toolConfigStackwidget.setCurrentWidget(self.penConfigPanel)  # 显示画笔配置面板
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setPenCursor()

    def on_use_recttool(self) -> None:
        """切换到矩形工具"""
        if self.currentTool is not None and self.currentTool != self.rectTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.rectTool
        pcfg.drawpanel.current_tool = ImageEditMode.RectTool
        self.toolConfigStackwidget.setCurrentWidget(self.rectPanel)  # 显示矩形配置面板
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.canvas.image_edit_mode = ImageEditMode.RectTool
        self.setCrossCursor()  # 设置十字光标

    def set_config(self, config: DrawPanelConfig):
        """加载配置设置"""
        # 画笔设置
        self.setPenToolWidth(config.pentool_width)
        self.setPenToolColor(config.pentool_color)
        self.penConfigPanel.thicknessSlider.setValue(int(config.pentool_width))
        self.penConfigPanel.shapeCombobox.setCurrentIndex(config.pentool_shape)
        
        # 修复工具设置
        self.setInpaintToolWidth(config.inpainter_width)
        self.inpaintConfigPanel.thicknessSlider.setValue(int(config.inpainter_width))
        self.inpaintConfigPanel.shapeCombobox.setCurrentIndex(config.inpainter_shape)
        
        # 矩形工具设置
        self.rectPanel.dilate_slider.setValue(config.recttool_dilate_ksize)
        self.rectPanel.autoChecker.setChecked(config.rectool_auto)
        self.rectPanel.methodComboBox.setCurrentIndex(config.rectool_method)
        
        # 设置当前工具
        if config.current_tool == ImageEditMode.HandTool:
            self.handTool.setChecked(True)
        elif config.current_tool == ImageEditMode.InpaintTool:
            self.inpaintTool.setChecked(True)
        elif config.current_tool == ImageEditMode.PenTool:
            self.penTool.setChecked(True)
        elif config.current_tool == ImageEditMode.RectTool:
            self.rectTool.setChecked(True)

    def get_pen_cursor(self, pen_color: QColor = None, pen_size = None, draw_shape=True, shape=PenShape.Circle) -> QCursor:
        """创建自定义画笔光标"""
        cross_size = 31  # 十字大小
        cross_len = cross_size // 4
        thickness = 3  # 边框厚度
        
        # 默认使用画笔颜色和大小
        if pen_color is None:
            pen_color = self.pentool_pen.color()
        if pen_size is None:
            pen_size = self.pentool_pen.width()
        
        # 根据缩放因子调整大小
        pen_size *= self.canvas.scale_factor
        map_size = max(cross_size+7, pen_size)  # 光标图像大小
        cursor_center = map_size // 2  # 中心点
        pen_radius = pen_size // 2  # 画笔半径
        pen_color.setAlpha(127)  # 设置半透明
        
        # 创建画笔
        pen = QPen(pen_color, thickness, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        pen.setDashPattern([3, 6])  # 虚线样式
        
        # 小尺寸时使用实线
        if pen_size < 20:
            pen.setStyle(Qt.PenStyle.SolidLine)

        # 创建光标图像
        cur_pixmap = QPixmap(QSizeF(map_size, map_size).toSize())
        cur_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cur_pixmap)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # 抗锯齿
        
        # 绘制形状
        if draw_shape:
            shape_rect = QRectF(cursor_center-pen_radius + thickness, 
                                cursor_center-pen_radius + thickness, 
                                pen_size - 2*thickness, 
                                pen_size - 2*thickness)
            if shape == PenShape.Circle:  # 圆形
                painter.drawEllipse(shape_rect)
            elif shape == PenShape.Rectangle:  # 矩形
                painter.drawRect(shape_rect)
            else:
                raise NotImplementedError
        
        # 绘制十字准线
        cross_left = (map_size  - 1 - cross_size) // 2 
        cross_right = map_size - cross_left

        # 绘制白色十字
        pen = QPen(Qt.GlobalColor.white, 5, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        cross_hline0 = QLineF(cross_left, cursor_center, cross_left+cross_len, cursor_center)
        cross_hline1 = QLineF(cross_right-cross_len, cursor_center, cross_right, cursor_center)
        cross_vline0 = QLineF(cursor_center, cross_left, cursor_center, cross_left+cross_len)
        cross_vline1 = QLineF(cursor_center, cross_right-cross_len, cursor_center, cross_right)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        
        # 绘制黑色十字（轮廓）
        pen.setWidth(3)
        pen.setColor(Qt.GlobalColor.black)
        painter.setPen(pen)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        painter.end()
        
        return QCursor(cur_pixmap)

    def on_incre_pensize(self):
        """增加画笔大小"""
        self.scalePen(1.1)

    def on_decre_pensize(self):
        """减小画笔大小"""
        self.scalePen(0.9)

    def scalePen(self, scale_factor):
        """缩放当前工具的画笔大小"""
        if self.currentTool == self.penTool:  # 画笔工具
            val = self.pentool_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            self.penConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setPenToolWidth(self.penConfigPanel.thicknessSlider.value())

        elif self.currentTool == self.inpaintTool:  # 修复工具
            val = self.inpaint_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            self.inpaintConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setInpaintToolWidth(self.inpaintConfigPanel.thicknessSlider.value())

    def showEvent(self, event) -> None:
        """显示事件处理"""
        if self.currentTool is not None:
            self.currentTool.setChecked(False)
            self.currentTool.setChecked(True)
        return super().showEvent(event)

    def on_finish_painting(self, stroke_item: StrokeImgItem):
        """完成绘制时的处理"""
        stroke_item.finishPainting()
        if not self.canvas.imgtrans_proj.img_valid:
            self.canvas.removeItem(stroke_item)
            return
        
        # 画笔工具处理
        if self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if rect is not None:
                # 创建撤销命令
                self.canvas.push_undo_command(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg))
            self.canvas.removeItem(stroke_item)
        
        # 修复工具处理
        elif self.currentTool == self.inpaintTool:
            self.inpaint_stroke = stroke_item
            if self.canvas.gv.ctrl_pressed:  # 按住Ctrl时不立即修复
                return
            else:
                self.runInpaint()  # 执行修复

    def on_finish_erasing(self, stroke_item: StrokeImgItem):
        """完成擦除时的处理"""
        stroke_item.finishPainting()
        
        # 修复工具的擦除逻辑
        if self.currentTool == self.inpaintTool:
            rect, mask, _ = stroke_item.clip(mask_only=True)
            if mask is None:
                self.canvas.removeItem(stroke_item)
                return
            mask = 255 - mask  # 反转遮罩
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            
            # 获取原始图像和修复区域
            origin = self.canvas.imgtrans_proj.img_array
            origin = origin[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpainted = self.canvas.imgtrans_proj.inpainted_array
            inpainted = inpainted[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_mask = self.canvas.imgtrans_proj.mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            
            # 如果没有需要擦除的内容
            if inpaint_mask.sum() == 0:
                self.canvas.removeItem(stroke_item)
                return
                
            # 应用遮罩
            mask = cv2.bitwise_and(mask, inpaint_mask)
            inpaint_mask = np.zeros_like(inpainted)
            inpaint_mask[mask > 0] = 1
            # 混合原始图像和修复区域
            erased_img = inpaint_mask * inpainted + (1 - inpaint_mask) * origin
            # 创建撤销命令
            self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, erased_img, mask, inpaint_rect))
            self.canvas.removeItem(stroke_item)

        # 画笔工具的擦除逻辑
        elif self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if self.canvas.erase_img_key is not None:
                self.canvas.drawingLayer.removeQImage(self.canvas.erase_img_key)
                self.canvas.erase_img_key = None
                self.canvas.stroke_img_item = None
            if rect is not None:
                # 创建撤销命令
                self.canvas.push_undo_command(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg, True))

    def runInpaint(self, inpaint_dict=None):
        """执行修复操作"""
        # 如果没有传入修复字典，则使用当前笔划
        if inpaint_dict is None:
            if self.inpaint_stroke is None:
                return
            elif self.inpaint_stroke.parentItem() is None:
                logger.warning("inpainting goes wrong")
                self.clearInpaintItems()
                return
                
            # 获取笔划的遮罩和区域
            rect, mask, _ = self.inpaint_stroke.clip(mask_only=True)
            if mask is None:
                self.clearInpaintItems()
                return
            
            # 扩大修复区域以获得更好的结果
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            img = self.canvas.imgtrans_proj.inpainted_array
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            rect_enlarged = enlarge_window(inpaint_rect, img.shape[1], img.shape[0])
            top = mask_y - rect_enlarged[1]
            bottom = rect_enlarged[3] - inpaint_rect[3]
            left = mask_x - rect_enlarged[0]
            right = rect_enlarged[2] - inpaint_rect[2]

            # 扩展遮罩
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            inpaint_rect = rect_enlarged
            img = img[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_dict = {'img': img, 'mask': mask, 'inpaint_rect': inpaint_rect}

        # 禁用编辑模式，执行修复
        self.canvas.image_edit_mode = ImageEditMode.NONE
        self.module_manager.canvas_inpaint(inpaint_dict)

    def on_inpaint_finished(self, inpaint_dict):
        """修复完成时的处理"""
        inpainted = inpaint_dict['inpainted']  # 修复后的图像
        inpaint_rect = inpaint_dict['inpaint_rect']  # 修复区域
        mask_array = self.canvas.imgtrans_proj.mask_array
        # 合并新旧遮罩
        mask = cv2.bitwise_or(inpaint_dict['mask'], mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]])
        # 创建撤销命令
        self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, inpainted, mask, inpaint_rect))
        self.clearInpaintItems()  # 清理临时项

    def on_inpaint_failed(self):
        """修复失败时的处理"""
        if self.currentTool == self.inpaintTool and self.inpaint_stroke is not None:
            self.clearInpaintItems()

    def on_canvasctrl_released(self):
        """画布Ctrl键释放事件处理"""
        if self.isVisible() and self.currentTool == self.inpaintTool:
            self.runInpaint()

    def on_begin_scale_tool(self, pos: QPointF):
        """开始缩放工具（调整画笔大小）"""
        # 根据当前工具选择画笔
        if self.currentTool == self.penTool:
            circle_pen = QPen(self.pentool_pen)
        elif self.currentTool == self.inpaintTool:
            circle_pen = QPen(self.inpaint_pen)
        else:
            return
        
        # 计算画笔半径
        pen_radius = circle_pen.widthF() / 2 * self.canvas.scale_factor
        r, g, b, a = circle_pen.color().getRgb()

        # 配置缩放圆
        circle_pen.setWidth(3)
        circle_pen.setStyle(Qt.PenStyle.DashLine)
        circle_pen.setDashPattern([3, 6])
        self.scale_circle.setPen(circle_pen)
        self.scale_circle.setBrush(QBrush(QColor(r, g, b, 127)))  # 半透明填充
        self.scale_circle.setPos(pos - QPointF(pen_radius, pen_radius))
        pen_size = 2 * pen_radius
        self.scale_circle.setRect(0, 0, pen_size, pen_size)
        self.scale_tool_pos = pos - QPointF(pen_size, pen_size)
        self.canvas.addItem(self.scale_circle)  # 添加到画布
        self.setCrossCursor()  # 设置十字光标

    def setCrossCursor(self):
        """设置十字光标（不带形状）"""
        self.canvas.gv.setCursor(self.get_pen_cursor(draw_shape=False))

    def on_scale_tool(self, pos: QPointF):
        """缩放工具调整中"""
        if self.scale_tool_pos is None:
            return
        # 计算新半径（限制在最小最大值之间）
        radius = pos.x() - self.scale_tool_pos.x()
        radius = max(min(radius, MAX_PEN_SIZE * self.canvas.scale_factor), MIN_PEN_SIZE * self.canvas.scale_factor)
        self.scale_circle.setRect(0, 0, radius, radius)  # 更新圆的大小

    def on_end_scale_tool(self):
        """结束缩放工具"""
        # 计算实际画笔大小
        circle_size = int(self.scale_circle.rect().width() / self.canvas.scale_factor)
        self.scale_tool_pos = None
        self.canvas.removeItem(self.scale_circle)  # 从画布移除

        # 更新画笔大小
        if self.currentTool == self.penTool:
            self.setPenToolWidth(circle_size)
            self.penConfigPanel.thicknessSlider.setValue(circle_size)
            self.setPenCursor()  # 更新光标
        elif self.currentTool == self.inpaintTool:
            self.setInpaintToolWidth(circle_size)
            self.inpaintConfigPanel.thicknessSlider.setValue(circle_size)
            self.setInpaintCursor()  # 更新光标

    def on_canvas_scalefactor_changed(self):
        """画布缩放因子变化时更新光标"""
        if not self.isVisible():
            return
        if self.currentTool == self.penTool:
            self.setPenCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintCursor()

    def setPenCursor(self):
        """设置画笔光标"""
        self.canvas.gv.setCursor(self.get_pen_cursor(shape=self.penConfigPanel.shape))

    def setInpaintCursor(self):
        """设置修复工具光标"""
        self.canvas.gv.setCursor(self.get_pen_cursor(INPAINT_BRUSH_COLOR, self.inpaint_pen.width(), shape=self.inpaintConfigPanel.shape))

    def on_handchecker_changed(self):
        """手形工具状态变化处理"""
        if self.handTool.isChecked():
            self.toolConfigStackwidget.hide()  # 隐藏配置面板
        else:
            self.toolConfigStackwidget.show()  # 显示配置面板

    def on_end_create_rect(self, rect: QRectF, mode: int):
        """完成创建矩形时的处理"""
        if self.currentTool == self.rectTool:
            self.canvas.image_edit_mode = ImageEditMode.NONE
            img = self.canvas.imgtrans_proj.inpainted_array
            im_h, im_w = img.shape[:2]

            # 获取矩形坐标并裁剪到图像范围内
            xyxy = [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]
            xyxy = np.array(xyxy)
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, im_w - 1)
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, im_h - 1)
            x1, y1, x2, y2 = xyxy.astype(np.int64)
            
            # 忽略无效矩形
            if y2 - y1 < 2 or x2 - x1 < 2:
                self.canvas.image_edit_mode = ImageEditMode.RectTool
                return
            
            # 修复模式
            if mode == 0:
                im = np.copy(img[y1: y2, x1: x2])  # 复制矩形区域图像
                # 获取遮罩分割方法
                maskseg_method = get_maskseg_method()
                # 生成遮罩
                inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=self.canvas.imgtrans_proj.mask_array[y1: y2, x1: x2])
                mask = self.rectPanel.post_process_mask(inpaint_mask_array)  # 后处理遮罩

                bground_rgb = bub_dict['bground_rgb']  # 背景颜色
                need_inpaint = bub_dict['need_inpaint']  # 是否需要修复

                # 创建修复字典
                inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2]}
                inpaint_dict['need_inpaint'] = need_inpaint
                inpaint_dict['bground_rgb'] = bground_rgb
                inpaint_dict['ballon_mask'] = ballon_mask
                
                # 创建用户预览遮罩
                user_preview_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                user_preview_mask[:, :, [0, 2, 3]] = (mask[:, :, np.newaxis] / 2).astype(np.uint8)
                self.inpaint_mask_item.setPixmap(ndarray2pixmap(user_preview_mask))
                self.inpaint_mask_item.setParentItem(self.canvas.baseLayer)
                self.inpaint_mask_item.setPos(x1, y1)
                
                # 根据自动设置决定是否立即修复
                if self.rectPanel.auto():
                    self.inpaintRect(inpaint_dict)
                else:
                    self.inpaint_mask_array = inpaint_mask_array
                    self.rect_inpaint_dict = inpaint_dict
            else:  # 擦除模式
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                erased = self.canvas.imgtrans_proj.img_array[y1: y2, x1: x2]  # 原始图像区域
                # 创建撤销命令
                self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, erased, mask, [x1, y1, x2, y2]))
                self.canvas.image_edit_mode = ImageEditMode.RectTool
            self.setCrossCursor()

    def inpaintRect(self, inpaint_dict):
        """执行矩形区域的修复"""
        img = inpaint_dict['img']
        mask = inpaint_dict['mask']
        need_inpaint = inpaint_dict['need_inpaint']
        bground_rgb = inpaint_dict['bground_rgb']
        ballon_mask = inpaint_dict['ballon_mask']
        
        # 如果不需要修复且配置要求检查
        if not need_inpaint and pcfg.module.check_need_inpaint:
            # 直接应用背景颜色
            img[np.where(ballon_mask > 0)] = bground_rgb
            # 创建撤销命令
            self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, img, mask, inpaint_dict['inpaint_rect'], merge_existing_mask=True))
            self.clearInpaintItems()
        else:
            self.runInpaint(inpaint_dict=inpaint_dict)  # 执行修复

    def on_rect_inpaintbtn_clicked(self):
        """矩形修复按钮点击事件"""
        if self.rect_inpaint_dict is not None:
            self.inpaintRect(self.rect_inpaint_dict)

    def on_rect_deletebtn_clicked(self):
        """矩形删除按钮点击事件"""
        self.clearInpaintItems()

    def on_rectool_ksize_changed(self):
        """矩形工具膨胀参数变化处理"""
        pcfg.drawpanel.recttool_dilate_ksize = self.rectPanel.dilate_slider.value()
        if self.currentTool != self.rectTool or self.inpaint_mask_array is None or self.inpaint_mask_item is None:
            return
        
        # 重新处理遮罩并更新预览
        mask = self.rectPanel.post_process_mask(self.inpaint_mask_array)
        self.rect_inpaint_dict['mask'] = mask
        user_preview_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        user_preview_mask[:, :, [0, 2, 3]] = (mask[:, :, np.newaxis] / 2).astype(np.uint8)
        self.inpaint_mask_item.setPixmap(ndarray2pixmap(user_preview_mask))

    def on_rectchecker_changed(self):
        """矩形工具状态变化处理"""
        if not self.rectTool.isChecked():
            self.clearInpaintItems()

    def hideEvent(self, e) -> None:
        """隐藏事件处理"""
        self.clearInpaintItems()
        return super().hideEvent(e)

    def clearInpaintItems(self):
        """清理所有修复相关的临时项"""
        self.rect_inpaint_dict = None
        self.inpaint_mask_array = None
        
        # 移除遮罩项
        if self.inpaint_mask_item is not None:
            if self.inpaint_mask_item.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_mask_item)
            if self.rectTool.isChecked():
                self.canvas.image_edit_mode = ImageEditMode.RectTool
                
        # 移除修复笔划
        if self.inpaint_stroke is not None:
            if self.inpaint_stroke.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_stroke)
            self.inpaint_stroke = None
            if self.inpaintTool.isChecked():
                self.canvas.image_edit_mode = ImageEditMode.InpaintTool

    def handle_page_changed(self):
        """页面变化时清理临时项"""
        self.clearInpaintItems()