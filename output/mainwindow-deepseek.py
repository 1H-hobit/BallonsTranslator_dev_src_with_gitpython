from typing import List, Union
import os, re, traceback, sys
import subprocess
import os.path as osp
from pathlib import Path
from functools import partial
import time

# GUI 组件
from tqdm import tqdm
from qtpy.QtWidgets import QAction, QFileDialog, QMenu, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSplitter, QListWidget, QShortcut, QListWidgetItem, QMessageBox, QTextEdit, QPlainTextEdit
from qtpy.QtCore import Qt, QPoint, QSize, QEvent, Signal
from qtpy.QtGui import QContextMenuEvent, QTextCursor, QGuiApplication, QIcon, QCloseEvent, QKeySequence, QKeyEvent, QPainter, QClipboard

# 工具类和模块
from utils.logger import logger as LOGGER
from utils.text_processing import is_cjk, full_len, half_len  # 文本处理工具
from utils.textblock import TextBlock, TextAlignment  # 文本块定义
from utils import shared  # 共享状态和配置
from utils.message import create_error_dialog, create_info_dialog  # 对话框工具
from modules.translators.trans_chatgpt import GPTTranslator  # ChatGPT翻译器
from modules import GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_TRANSLATORS, GET_VALID_OCR  # 模块获取函数

# 自定义组件
from .misc import parse_stylesheet, set_html_family, QKEY  # 样式工具
from utils.config import ProgramConfig, pcfg, save_config, text_styles, save_text_styles, load_textstyle_from, FontFormat  # 配置管理
from utils.proj_imgtrans import ProjImgTrans  # 图像翻译项目
from .canvas import Canvas  # 画布组件
from .configpanel import ConfigPanel  # 配置面板
from .module_manager import ModuleManager  # 模块管理器
from .textedit_area import SourceTextEdit, SelectTextMiniMenu, TransTextEdit  # 文本编辑区域
from .drawingpanel import DrawingPanel  # 绘图面板
from .scenetext_manager import SceneTextManager, TextPanel, PasteSrcItemsCommand  # 场景文本管理
from .mainwindowbars import TitleBar, LeftBar, BottomBar  # 窗口标题栏/侧边栏/底部栏
from .io_thread import ImgSaveThread, ImportDocThread, ExportDocThread  # IO线程
from .custom_widget import Widget, ViewWidget  # 自定义控件
from .global_search_widget import GlobalSearchWidget  # 全局搜索组件
from .textedit_commands import GlobalRepalceAllCommand  # 文本编辑命令
from .framelesswindow import FramelessWindow  # 无边框窗口
from .drawing_commands import RunBlkTransCommand  # 绘图命令
from .keywordsubwidget import KeywordSubWidget  # 关键词替换组件
from . import shared_widget as SW  # 共享组件
from .custom_widget import MessageBox, FrameLessMessageBox, ImgtransProgressMessageBox  # 自定义对话框

class PageListView(QListWidget):
    """页面列表视图组件"""
    reveal_file = Signal()  # 显示文件信号

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 设置缩略图大小
        self.setIconSize(QSize(shared.PAGELIST_THUMBNAIL_SIZE, shared.PAGELIST_THUMBNAIL_SIZE))

    def contextMenuEvent(self, e: QContextMenuEvent):
        """上下文菜单事件"""
        menu = QMenu()
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))  # 在文件资源管理器中显示
        rst = menu.exec_(e.globalPos())

        if rst == reveal_act:
            self.reveal_file.emit()  # 触发显示文件信号

        return super().contextMenuEvent(e)

# 根据运行模式选择窗口基类
mainwindow_cls = Widget if shared.HEADLESS else FramelessWindow

class MainWindow(mainwindow_cls):
    """主窗口类"""
    imgtrans_proj: ProjImgTrans = ProjImgTrans()  # 图像翻译项目实例
    save_on_page_changed = True  # 页面切换时自动保存
    opening_dir = False  # 是否正在打开目录
    page_changing = False  # 是否正在切换页面
    postprocess_mt_toggle = True  # 是否启用翻译后处理
    translator = None  # 翻译器实例

    # 信号定义
    restart_signal = Signal()
    create_errdialog = Signal(str, str, str)
    create_infodialog = Signal(dict)
    
    def __init__(self, app: QApplication, config: ProgramConfig, open_dir='', **exec_args) -> None:
        super().__init__()
        # 设置错误/信息对话框信号
        shared.create_errdialog_in_mainthread = self.create_errdialog.emit
        self.create_errdialog.connect(self.on_create_errdialog)
        shared.create_infodialog_in_mainthread = self.create_infodialog.emit
        self.create_infodialog.connect(self.on_create_infodialog)
        shared.register_view_widget = self.register_view_widget

        self.app = app
        self.backup_blkstyles = []  # 文本样式备份
        self._run_imgtrans_wo_textstyle_update = False  # 是否不更新文本样式运行翻译

        # 初始化方法调用
        self.setupThread()
        self.setupUi()
        self.setupConfig()
        self.setupShortcuts()
        self.setupRegisterWidget()
        self.showMaximized()  # 最大化显示
        self.setAcceptDrops(True)  # 接受拖放
        self.module_manager.inpaint_thread.inpainter.main_window = self  # 设置修复模块的主窗口引用

        # 打开项目或最近项目
        if open_dir != '' and osp.exists(open_dir):
            self.OpenProj(open_dir)
        elif pcfg.open_recent_on_startup:
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.OpenProj(proj_dir)

        # 无头模式下的批处理
        if shared.HEADLESS:
            self.run_batch(**exec_args)

        # macOS特定设置
        if shared.ON_MACOS:
            self.hideSystemTitleBar()
            self.showMaximized()

    # ====================== 核心方法 ======================
    def setupThread(self):
        """初始化后台线程"""
        self.imsave_thread = ImgSaveThread()  # 图片保存线程
        self.export_doc_thread = ExportDocThread()  # 导出文档线程
        self.export_doc_thread.fin_io.connect(self.on_fin_export_doc)
        self.import_doc_thread = ImportDocThread(self)  # 导入文档线程
        self.import_doc_thread.fin_io.connect(self.on_fin_import_doc)

    def setupUi(self):
        """初始化用户界面"""
        # 基本窗口设置
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)
        
        # 配置面板
        self.configPanel = ConfigPanel(self)
        self.configPanel.trans_config_panel.show_pre_MT_keyword_window.connect(self.show_pre_MT_keyword_window)
        self.configPanel.trans_config_panel.show_MT_keyword_window.connect(self.show_MT_keyword_window)
        self.configPanel.trans_config_panel.show_OCR_keyword_window.connect(self.show_OCR_keyword_window)

        # 左侧边栏
        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        self.leftBar.globalSearchChecker.clicked.connect(self.on_set_gsearch_widget)
        self.leftBar.open_dir.connect(self.OpenProj)
        self.leftBar.open_json_proj.connect(self.openJsonProj)
        self.leftBar.save_proj.connect(self.manual_save)
        self.leftBar.export_doc.connect(self.on_export_doc)
        self.leftBar.import_doc.connect(self.on_import_doc)
        self.leftBar.export_src_txt.connect(lambda : self.on_export_txt(dump_target='source'))
        self.leftBar.export_trans_txt.connect(lambda : self.on_export_txt(dump_target='translation'))
        self.leftBar.export_src_md.connect(lambda : self.on_export_txt(dump_target='source', suffix='.md'))
        self.leftBar.export_trans_md.connect(lambda : self.on_export_txt(dump_target='translation', suffix='.md'))
        self.leftBar.import_trans_txt.connect(self.on_import_trans_txt)

        # 页面列表
        self.pageList = PageListView()
        self.pageList.reveal_file.connect(self.on_reveal_file)
        self.pageList.setHidden(True)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)

        # 左侧堆叠组件
        self.leftStackWidget = QStackedWidget(self)
        self.leftStackWidget.addWidget(self.pageList)

        # 全局搜索组件
        self.global_search_widget = GlobalSearchWidget(self.leftStackWidget)
        self.global_search_widget.req_update_pagetext.connect(self.on_req_update_pagetext)
        self.global_search_widget.req_move_page.connect(self.on_req_move_page)
        self.imsave_thread.img_writed.connect(self.global_search_widget.on_img_writed)
        self.global_search_widget.search_tree.result_item_clicked.connect(self.on_search_result_item_clicked)
        self.leftStackWidget.addWidget(self.global_search_widget)
        
        # 中央堆叠组件
        self.centralStackWidget = QStackedWidget(self)
        
        # 标题栏和底部栏
        self.titleBar = TitleBar(self)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        self.titleBar.display_lang_changed.connect(self.on_display_lang_changed)
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)

        # 主布局
        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)
        mainHLayout.setSpacing(0)

        # 画布组件
        SW.canvas = self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.textstack_changed.connect(self.on_textstack_changed)
        self.canvas.run_blktrans.connect(self.on_run_blktrans)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)
        self.canvas.originallayer_trans_slider = self.bottomBar.originalSlider
        self.canvas.textlayer_trans_slider = self.bottomBar.textlayerSlider
        self.canvas.copy_src_signal.connect(self.on_copy_src)
        self.canvas.paste_src_signal.connect(self.on_paste_src)

        # 连接滑块事件
        self.bottomBar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        self.bottomBar.textlayerSlider.valueChanged.connect(self.canvas.setTextLayerTransparencyBySlider)
        
        # 绘图面板和文本面板
        self.drawingPanel = DrawingPanel(self.canvas, self.configPanel.inpaint_config_panel)
        self.textPanel = TextPanel(self.app)
        self.textPanel.formatpanel.foldTextBtn.checkStateChanged.connect(self.fold_textarea)
        self.textPanel.formatpanel.sourceBtn.checkStateChanged.connect(self.show_source_text)
        self.textPanel.formatpanel.transBtn.checkStateChanged.connect(self.show_trans_text)
        self.textPanel.formatpanel.textstyle_panel.export_style.connect(self.export_tstyles)
        self.textPanel.formatpanel.textstyle_panel.import_style.connect(self.import_tstyles)

        # 关键词替换窗口
        self.ocrSubWidget = KeywordSubWidget(self.tr("Keyword substitution for source text"))
        self.ocrSubWidget.setParent(self)
        self.ocrSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.ocrSubWidget.hide()
        self.mtPreSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation source text"))
        self.mtPreSubWidget.setParent(self)
        self.mtPreSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtPreSubWidget.hide()
        self.mtSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation"))
        self.mtSubWidget.setParent(self)
        self.mtSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtSubWidget.hide()

        # 场景文本管理器
        SW.st_manager = self.st_manager = SceneTextManager(self.app, self, self.canvas, self.textPanel)
        self.st_manager.new_textblk.connect(self.canvas.search_widget.on_new_textblk)
        self.canvas.search_widget.pairwidget_list = self.st_manager.pairwidget_list
        self.canvas.search_widget.textblk_item_list = self.st_manager.textblk_item_list
        self.canvas.search_widget.replace_one.connect(self.st_manager.on_page_replace_one)
        self.canvas.search_widget.replace_all.connect(self.st_manager.on_page_replace_all)

        # 漫画翻译面板布局
        self.rightComicTransStackPanel = QStackedWidget(self)
        self.rightComicTransStackPanel.addWidget(self.drawingPanel)
        self.rightComicTransStackPanel.addWidget(self.textPanel)
        self.rightComicTransStackPanel.currentChanged.connect(self.on_transpanel_changed)

        self.comicTransSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.comicTransSplitter.addWidget(self.leftStackWidget)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)

        self.centralStackWidget.addWidget(self.comicTransSplitter)
        self.centralStackWidget.addWidget(self.configPanel)

        # 文本选择小菜单
        self.selectext_minimenu = self.st_manager.selectext_minimenu = SelectTextMiniMenu(self.app, self)
        self.selectext_minimenu.block_current_editor.connect(self.st_manager.on_block_current_editor)
        self.selectext_minimenu.hide()

        # 主垂直布局
        mainVBoxLayout = QVBoxLayout(self)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        margin = mainVBoxLayout.contentsMargins()
        self.main_margin = margin
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)

        self.mainvlayout = mainVBoxLayout
        self.comicTransSplitter.setStretchFactor(0, 1)
        self.comicTransSplitter.setStretchFactor(1, 10)
        self.comicTransSplitter.setStretchFactor(2, 1)
        self.imgtrans_progress_msgbox = ImgtransProgressMessageBox()
        self.resetStyleSheet()  # 应用样式表

    # ====================== 模块管理方法 ======================
    def on_finish_setdetector(self):
        """文本检测器设置完成"""
        module_manager = self.module_manager
        if module_manager.textdetector is not None:
            name = module_manager.textdetector.name
            pcfg.module.textdetector = name
            self.configPanel.detect_config_panel.setDetector(name)
            self.bottomBar.textdet_selector.setSelectedValue(name)
            LOGGER.info('Text detector set to {}'.format(name))

    def on_finish_setocr(self):
        """OCR设置完成"""
        module_manager = self.module_manager
        if module_manager.ocr is not None:
            name = module_manager.ocr.name
            pcfg.module.ocr = name
            self.configPanel.ocr_config_panel.setOCR(name)
            self.bottomBar.ocr_selector.setSelectedValue(name)
            LOGGER.info('OCR set to {}'.format(name))

    def on_finish_setinpainter(self):
        """修复模块设置完成"""
        module_manager = self.module_manager
        if module_manager.inpainter is not None:
            name = module_manager.inpainter.name
            pcfg.module.inpainter = name
            self.configPanel.inpaint_config_panel.setInpainter(name)
            self.bottomBar.inpaint_selector.setSelectedValue(name)
            LOGGER.info('Inpainter set to {}'.format(name))

    def on_finish_settranslator(self):
        """翻译器设置完成"""
        module_manager = self.module_manager
        translator = module_manager.translator
        if translator is not None:
            name = translator.name
            pcfg.module.translator = name
            self.bottomBar.trans_selector.finishSetTranslator(translator)
            self.configPanel.trans_config_panel.finishSetTranslator(translator)
            LOGGER.info('Translator set to {}'.format(name))
        else:
            LOGGER.error('invalid translator')
        
    def on_enable_module(self, idx, checked):
        """启用/禁用模块"""
        if idx == 0:  # 文本检测
            pcfg.module.enable_detect = checked
            self.bottomBar.textdet_selector.setVisible(checked)
        elif idx == 1:  # OCR
            pcfg.module.enable_ocr = checked
            self.bottomBar.ocr_selector.setVisible(checked)
        elif idx == 2:  # 翻译
            pcfg.module.enable_translate = checked
            self.bottomBar.trans_selector.setVisible(checked)
        elif idx == 3:  # 修复
            pcfg.module.enable_inpaint = checked
            self.bottomBar.inpaint_selector.setVisible(checked)

    # ====================== 配置方法 ======================
    def setupConfig(self):
        """初始化配置"""
        # 透明度设置
        self.bottomBar.originalSlider.setValue(int(pcfg.original_transparency * 100))
        
        # 模块选择器初始化
        self.bottomBar.trans_selector.selector.addItems(GET_VALID_TRANSLATORS())
        self.bottomBar.ocr_selector.selector.addItems(GET_VALID_OCR())
        # 特殊处理InternVL3-8B模型
        if 'InternVL3-8B' not in GET_VALID_OCR():
                GET_VALID_OCR().append('InternVL3-8B')
                self.bottomBar.ocr_selector.selector.addItem('InternVL3-8B')
        self.bottomBar.textdet_selector.selector.addItems(GET_VALID_TEXTDETECTORS())
        self.bottomBar.textdet_selector.selector.currentTextChanged.connect(self.on_textdet_changed)
        self.bottomBar.inpaint_selector.selector.addItems(GET_VALID_INPAINTERS())
        self.bottomBar.inpaint_selector.selector.currentTextChanged.connect(self.on_inpaint_changed)
        self.bottomBar.trans_selector.cfg_clicked.connect(self.to_trans_config)
        self.bottomBar.trans_selector.selector.currentTextChanged.connect(self.on_trans_changed)
        self.bottomBar.trans_selector.tgt_selector.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.bottomBar.trans_selector.src_selector.currentTextChanged.connect(self.on_trans_src_changed)
        self.bottomBar.textdet_selector.cfg_clicked.connect(self.to_detect_config)
        self.bottomBar.inpaint_selector.cfg_clicked.connect(self.to_inpaint_config)
        self.bottomBar.ocr_selector.cfg_clicked.connect(self.to_ocr_config)
        self.bottomBar.ocr_selector.selector.currentTextChanged.connect(self.on_ocr_changed)
        
        # 根据配置显示模块
        self.bottomBar.textdet_selector.setVisible(pcfg.module.enable_detect)
        self.bottomBar.ocr_selector.setVisible(pcfg.module.enable_ocr)
        self.bottomBar.trans_selector.setVisible(pcfg.module.enable_translate)
        self.bottomBar.inpaint_selector.setVisible(pcfg.module.enable_inpaint)

        # 翻译配置连接
        self.configPanel.trans_config_panel.target_combobox.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.configPanel.trans_config_panel.source_combobox.currentTextChanged.connect(self.on_trans_src_changed)

        # 其他UI设置
        self.drawingPanel.maskTransperancySlider.setValue(int(pcfg.mask_transparency * 100))
        self.leftBar.initRecentProjMenu(pcfg.recent_proj_list)
        self.leftBar.showPageListLabel.setChecked(pcfg.show_page_list)
        self.updatePageList()
        self.leftBar.save_config.connect(self.save_config)
        self.leftBar.imgTransChecker.setChecked(True)
        self.st_manager.formatpanel.global_format = pcfg.global_fontformat
        self.st_manager.formatpanel.set_active_format(pcfg.global_fontformat)
        
        # 面板初始状态
        self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)
        self.st_manager.formatpanel.foldTextBtn.setChecked(pcfg.fold_textarea)
        self.st_manager.formatpanel.transBtn.setCheckState(pcfg.show_trans_text)
        self.st_manager.formatpanel.sourceBtn.setCheckState(pcfg.show_source_text)
        self.fold_textarea(pcfg.fold_textarea)
        self.show_trans_text(pcfg.show_trans_text)
        self.show_source_text(pcfg.show_source_text)

        # 模块管理器初始化
        self.module_manager = module_manager = ModuleManager(self.imgtrans_proj)
        module_manager.finish_translate_page.connect(self.finishTranslatePage)
        module_manager.imgtrans_pipeline_finished.connect(self.on_imgtrans_pipeline_finished)
        module_manager.page_trans_finished.connect(self.on_pagtrans_finished)
        module_manager.setupThread(self.configPanel, self.imgtrans_progress_msgbox, self.ocr_postprocess, self.translate_preprocess, self.translate_postprocess)
        module_manager.progress_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)
        module_manager.blktrans_pipeline_finished.connect(self.on_blktrans_finished)
        module_manager.imgtrans_thread.post_process_mask = self.drawingPanel.rectPanel.post_process_mask
        module_manager.inpaint_thread.finish_set_module.connect(self.on_finish_setinpainter)
        module_manager.translate_thread.finish_set_module.connect(self.on_finish_settranslator)
        module_manager.textdetect_thread.finish_set_module.connect(self.on_finish_setdetector)
        module_manager.ocr_thread.finish_set_module.connect(self.on_finish_setocr)
        
        # 初始模块设置
        module_manager.setTextDetector()
        module_manager.setOCR()
        module_manager.setTranslator()
        module_manager.setInpainter()

        # 连接运行翻译信号
        self.leftBar.run_imgtrans_clicked.connect(self.run_imgtrans)

        # 暗黑模式设置
        self.titleBar.darkModeAction.setChecked(pcfg.darkmode)

        # 绘图面板初始化
        self.drawingPanel.set_config(pcfg.drawpanel)
        self.drawingPanel.initDLModule(module_manager)

        # 全局搜索设置
        self.global_search_widget.imgtrans_proj = self.imgtrans_proj
        self.global_search_widget.setupReplaceThread(self.st_manager.pairwidget_list, self.st_manager.textblk_item_list)
        self.global_search_widget.replace_thread.finished.connect(self.on_global_replace_finished)

        # 配置面板初始化
        self.configPanel.setupConfig()
        self.configPanel.save_config.connect(self.save_config)
        self.configPanel.reload_textstyle.connect(self.load_textstyle_from_proj_dir)
        self.configPanel.show_only_custom_font.connect(self.on_show_only_custom_font)
        if pcfg.let_show_only_custom_fonts_flag:
            self.on_show_only_custom_font(True)

        # 文本块模式设置
        textblock_mode = pcfg.imgtrans_textblock
        if pcfg.imgtrans_textedit:
            if textblock_mode:
                self.bottomBar.textblockChecker.setChecked(True)
            self.bottomBar.texteditChecker.click()
        elif pcfg.imgtrans_paintmode:
            self.bottomBar.paintChecker.click()

        # 文本样式面板初始化
        self.textPanel.formatpanel.textstyle_panel.initStyles(text_styles)

        # 搜索配置
        self.canvas.search_widget.whole_word_toggle.setChecked(pcfg.fsearch_whole_word)
        self.canvas.search_widget.case_sensitive_toggle.setChecked(pcfg.fsearch_case)
        self.canvas.search_widget.regex_toggle.setChecked(pcfg.fsearch_regex)
        self.canvas.search_widget.range_combobox.setCurrentIndex(pcfg.fsearch_range)
        self.global_search_widget.whole_word_toggle.setChecked(pcfg.gsearch_whole_word)
        self.global_search_widget.case_sensitive_toggle.setChecked(pcfg.gsearch_case)
        self.global_search_widget.regex_toggle.setChecked(pcfg.gsearch_regex)
        self.global_search_widget.range_combobox.setCurrentIndex(pcfg.gsearch_range)

        # 初始绘图模式
        if self.rightComicTransStackPanel.isHidden():
            self.setPaintMode()

        # 加载关键词替换配置
        try:
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.ocr_sublist = []
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)

        try:
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.pre_mt_sublist = []
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)

        try:
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.mt_sublist = []
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)

    # ====================== UI切换方法 ======================
    def setupImgTransUI(self):
        """切换到图像翻译UI"""
        self.centralStackWidget.setCurrentIndex(0)
        if self.leftBar.needleftStackWidget():
            self.leftStackWidget.show()
        else:
            self.leftStackWidget.hide()

    def setupConfigUI(self):
        """切换到配置UI"""
        self.centralStackWidget.setCurrentIndex(1)

    def set_display_lang(self, lang: str):
        """设置显示语言"""
        self.retranslateUI()

    # ====================== 项目管理方法 ======================
    def OpenProj(self, proj_path: str):
        """打开项目"""
        if osp.isdir(proj_path):
            self.openDir(proj_path)
        else:
            self.openJsonProj(proj_path)
        
        if pcfg.let_textstyle_indep_flag and not shared.HEADLESS:
            self.load_textstyle_from_proj_dir(from_proj=True)

    def load_textstyle_from_proj_dir(self, from_proj=False):
        """从项目目录加载文本样式"""
        if from_proj:
            text_style_path = osp.join(self.imgtrans_proj.directory, 'textstyles.json')
        else:
            text_style_path = 'config/textstyles/default.json'
        if osp.exists(text_style_path):
            load_textstyle_from(text_style_path)
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
        else:
            pcfg.text_styles_path = text_style_path
            save_text_styles()

    def on_show_only_custom_font(self, only_custom: bool):
        """是否只显示自定义字体"""
        if only_custom:
            font_list = shared.CUSTOM_FONTS
        else:
            font_list = shared.FONT_FAMILIES
        self.textPanel.formatpanel.familybox.update_font_list(font_list)

    def openDir(self, directory: str):
        """打开目录"""
        try:
            self.opening_dir = True
            self.imgtrans_proj.load(directory)
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(osp.basename(directory))
            self.updatePageList()
            self.opening_dir = False
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project ') + directory)
            return
        
    def dropOpenDir(self, directory: str):
        """拖放打开目录"""
        if isinstance(directory, str) and osp.exists(directory):
            self.leftBar.updateRecentProjList(directory)
            self.OpenProj(directory)

    def openJsonProj(self, json_path: str):
        """从JSON文件打开项目"""
        try:
            self.opening_dir = True
            self.imgtrans_proj.load_from_json(json_path)
            self.st_manager.clearSceneTextitems()
            self.leftBar.updateRecentProjList(self.imgtrans_proj.proj_path)
            self.updatePageList()
            self.titleBar.setTitleContent(osp.basename(self.imgtrans_proj.proj_path))
            self.opening_dir = False
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project from') + json_path)
        
    def updatePageList(self):
        """更新页面列表"""
        if self.pageList.count() != 0:
            self.pageList.clear()
        # 根据页面数量决定是否显示缩略图
        if len(self.imgtrans_proj.pages) >= shared.PAGELIST_THUMBNAIL_MAXNUM:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname:\
                QListWidgetItem(QIcon(osp.join(self.imgtrans_proj.directory, imgname)), imgname)
        # 添加页面项
        for imgname in self.imgtrans_proj.pages:
            lstitem =  item_func(imgname)
            self.pageList.addItem(lstitem)
            if imgname == self.imgtrans_proj.current_img:
                self.pageList.setCurrentItem(lstitem)

    def pageLabelStateChanged(self):
        """页面标签状态变化处理"""
        setup = self.leftBar.showPageListLabel.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            if self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.pageList)
        else:
            self.leftStackWidget.hide()
        pcfg.show_page_list = setup
        save_config()

    # ====================== 事件处理方法 ======================
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭事件处理"""
        self.st_manager.hovering_transwidget = None
        self.st_manager.blockSignals(True)
        self.canvas.prepareClose()
        self.save_config()
        if not self.imgtrans_proj.is_empty:
            self.imgtrans_proj.save()
        return super().closeEvent(event)

    def changeEvent(self, event: QEvent):
        """窗口状态变化事件"""
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                if not shared.ON_MACOS:
                    self.titleBar.maxBtn.setChecked(True)
        elif event.type() == QEvent.Type.ActivationChange:
            self.canvas.on_activation_changed()

        super().changeEvent(event)
    
    def retranslateUI(self):
        """重新翻译UI(需要重启)"""
        msg = QMessageBox()
        msg.setText(self.tr('Restart to apply changes? \n'))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.Yes:
            self.restart_signal.emit()

    def save_config(self):
        """保存配置"""
        save_config()

    def onHideCanvas(self):
        """隐藏画布处理"""
        self.canvas.alt_pressed = False
        self.canvas.scale_tool_mode = False

    def conditional_save(self):
        """条件保存"""
        if self.canvas.projstate_unsaved and not self.opening_dir:
            update_scene_text = save_proj = self.canvas.text_change_unsaved()
            save_rst_only = not self.canvas.draw_change_unsaved()
            if not save_rst_only:
                save_proj = True
            
            self.saveCurrentPage(update_scene_text, save_proj, restore_interface=True, save_rst_only=save_rst_only)

    def pageListCurrentItemChanged(self):
        """当前页面项变化处理"""
        item = self.pageList.currentItem()
        self.page_changing = True
        if item is not None:
            if self.save_on_page_changed:
                self.conditional_save()
            self.imgtrans_proj.set_current_img(item.text())
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
            self.module_manager.handle_page_changed()
            self.drawingPanel.handle_page_changed()
            
        self.page_changing = False

    # ====================== 快捷键设置 ======================
    def setupShortcuts(self):
        """设置快捷键"""
        # 标题栏信号连接
        self.titleBar.nextpage_trigger.connect(self.shortcutNext) 
        self.titleBar.prevpage_trigger.connect(self.shortcutBefore)
        self.titleBar.textedit_trigger.connect(self.shortcutTextedit)
        self.titleBar.drawboard_trigger.connect(self.shortcutDrawboard)
        self.titleBar.redo_trigger.connect(self.on_redo)
        self.titleBar.undo_trigger.connect(self.on_undo)
        self.titleBar.page_search_trigger.connect(self.on_page_search)
        self.titleBar.global_search_trigger.connect(self.on_global_search)
        self.titleBar.replacePreMTkeyword_trigger.connect(self.show_pre_MT_keyword_window)
        self.titleBar.replaceMTkeyword_trigger.connect(self.show_MT_keyword_window)
        self.titleBar.replaceOCRkeyword_trigger.connect(self.show_OCR_keyword_window)
        self.titleBar.run_trigger.connect(self.leftBar.runImgtransBtn.click)
        self.titleBar.run_woupdate_textstyle_trigger.connect(self.run_imgtrans_wo_textstyle_update)
        self.titleBar.translate_page_trigger.connect(self.on_transpagebtn_pressed)
        self.titleBar.enable_module.connect(self.on_enable_module)
        self.titleBar.importtstyle_trigger.connect(self.import_tstyles)
        self.titleBar.exporttstyle_trigger.connect(self.export_tstyles)
        self.titleBar.darkmode_trigger.connect(self.on_darkmode_triggered)

        # 向前翻页快捷键 (A/PageUp)
        shortcutA = QShortcut(QKeySequence("A"), self)
        shortcutA.activated.connect(self.shortcutBefore)
        shortcutPageUp = QShortcut(QKeySequence(QKeySequence.StandardKey.MoveToPreviousPage), self)
        shortcutPageUp.activated.connect(self.shortcutBefore)

        # 向后翻页快捷键 (D/PageDown)
        shortcutD = QShortcut(QKeySequence("D"), self)
        shortcutD.activated.connect(self.shortcutNext)
        shortcutPageDown = QShortcut(QKeySequence(QKeySequence.StandardKey.MoveToNextPage), self)
        shortcutPageDown.activated.connect(self.shortcutNext)

        # 其他功能快捷键
        shortcutTextblock = QShortcut(QKeySequence("W"), self)  # 文本块模式
        shortcutTextblock.activated.connect(self.shortcutTextblock)
        shortcutZoomIn = QShortcut(QKeySequence.StandardKey.ZoomIn, self)  # 放大
        shortcutZoomIn.activated.connect(self.canvas.gv.scale_up_signal)
        shortcutZoomOut = QShortcut(QKeySequence.StandardKey.ZoomOut, self)  # 缩小
        shortcutZoomOut.activated.connect(self.canvas.gv.scale_down_signal)
        shortcutCtrlD = QShortcut(QKeySequence("Ctrl+D"), self)  # 删除
        shortcutCtrlD.activated.connect(self.shortcutCtrlD)
        shortcutSpace = QShortcut(QKeySequence("Space"), self)  # 空格键
        shortcutSpace.activated.connect(self.shortcutSpace)
        shortcutSelectAll = QShortcut(QKeySequence.StandardKey.SelectAll, self)  # 全选
        shortcutSelectAll.activated.connect(self.shortcutSelectAll)

        shortcutEscape = QShortcut(QKeySequence("Escape"), self)  # 退出键
        shortcutEscape.activated.connect(self.shortcutEscape)

        # 文本格式快捷键
        shortcutBold = QShortcut(QKeySequence.StandardKey.Bold, self)  # 粗体
        shortcutBold.activated.connect(self.shortcutBold)
        shortcutItalic = QShortcut(QKeySequence.StandardKey.Italic, self)  # 斜体
        shortcutItalic.activated.connect(self.shortcutItalic)
        shortcutUnderline = QShortcut(QKeySequence.StandardKey.Underline, self)  # 下划线
        shortcutUnderline.activated.connect(self.shortcutUnderline)

        shortcutDelete = QShortcut(QKeySequence.StandardKey.Delete, self)  # 删除键
        shortcutDelete.activated.connect(self.shortcutDelete)

        # 绘图工具快捷键
        drawpanel_shortcuts = {'hand': 'H', 'rect': 'R', 'inpaint': 'J', 'pen': 'B'}
        for tool_name, shortcut_key in drawpanel_shortcuts.items():
            shortcut = QShortcut(QKeySequence(shortcut_key), self)
            shortcut.activated.connect(partial(self.drawingPanel.shortcutSetCurrentToolByName, tool_name))
            self.drawingPanel.setShortcutTip(tool_name, shortcut_key)

    # ====================== 快捷键功能实现 ======================
    def shortcutNext(self):
        """下一页/下一个文本项"""
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_D:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 0:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row + 1) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutBefore(self):
        """上一页/上一个文本项"""
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_A:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 0:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(-1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(-1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row - 1 + page_count) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutTextedit(self):
        """切换文本编辑模式"""
        if self.centralStackWidget.currentIndex() == 0:
            self.bottomBar.texteditChecker.click()

    def shortcutTextblock(self):
        """切换文本块模式"""
        if self.centralStackWidget.currentIndex() == 0:
            if self.bottomBar.texteditChecker.isChecked():
                self.bottomBar.textblockChecker.click()

    def shortcutDrawboard(self):
        """切换绘图板模式"""
        if self.centralStackWidget.currentIndex() == 0:
            self.bottomBar.paintChecker.click()

    def shortcutCtrlD(self):
        """删除快捷键"""
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.delete_btn.click()
            elif self.canvas.textEditMode():
                self.canvas.delete_textblks.emit(0)

    def shortcutSelectAll(self):
        """全选快捷键"""
        if self.centralStackWidget.currentIndex() == 0:
            if self.textPanel.isVisible():
                self.st_manager.set_blkitems_selection(True)

    def shortcutSpace(self):
        """空格键功能(修复)"""
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.inpaint_btn.click()

    def shortcutBold(self):
        """粗体快捷键"""
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.boldBtn.click()

    def shortcutDelete(self):
        """删除键功能"""
        if self.canvas.gv.isVisible():
            self.canvas.delete_textblks.emit(1)

    def shortcutItalic(self):
        """斜体快捷键"""
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.italicBtn.click()

    def shortcutUnderline(self):
        """下划线快捷键"""
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.underlineBtn.click()

    def on_redo(self):
        """重做操作"""
        self.canvas.redo()

    def on_undo(self):
        """撤销操作"""
        self.canvas.undo()

    def on_page_search(self):
        """页面内搜索"""
        if self.canvas.gv.isVisible():
            fo = self.app.focusObject()
            sel_text = ''
            tgt_edit = None
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
                tgt_edit = self.st_manager.pairwidget_list[blkitem.idx].e_trans
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
                if isinstance(fo, SourceTextEdit):
                    tgt_edit = fo
            se = self.canvas.search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)

            if self.canvas.search_widget.isHidden():
                self.canvas.search_widget.show()
            self.canvas.search_widget.setCurrentEditor(tgt_edit)

    def on_global_search(self):
        """全局搜索"""
        if self.canvas.gv.isVisible():
            if not self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.click()
            fo = self.app.focusObject()
            sel_text = ''
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
            se = self.global_search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)
                
                self.global_search_widget.commit_search()

    def show_pre_MT_keyword_window(self):
        """显示机器翻译前关键词替换窗口"""
        self.mtPreSubWidget.show()

    def show_MT_keyword_window(self):
        """显示机器翻译关键词替换窗口"""
        self.mtSubWidget.show()

    def show_OCR_keyword_window(self):
        """显示OCR关键词替换窗口"""
        self.ocrSubWidget.show()

    # ====================== 全局搜索相关方法 ======================
    def on_req_update_pagetext(self):
        """更新页面文本请求"""
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()

    def on_req_move_page(self, page_name: str, force_save=False):
        """移动页面请求"""
        ori_save = self.save_on_page_changed
        self.save_on_page_changed = False
        current_img = self.imgtrans_proj.current_img
        if current_img == page_name and not force_save:
            return
        if current_img not in self.global_search_widget.page_set:
            if self.canvas.projstate_unsaved: 
                self.saveCurrentPage()
        else:
            self.saveCurrentPage(save_rst_only=True)
        self.pageList.setCurrentRow(self.imgtrans_proj.pagename2idx(page_name))
        self.save_on_page_changed = ori_save

    def on_search_result_item_clicked(self, pagename: str, blk_idx: int, is_src: bool, start: int, end: int):
        """点击搜索结果项"""
        idx = self.imgtrans_proj.pagename2idx(pagename)
        self.pageList.setCurrentRow(idx)
        pw = self.st_manager.pairwidget_list[blk_idx]
        edit = pw.e_source if is_src else pw.e_trans
        edit.setFocus()
        edit.ensure_scene_visible.emit()
        cursor = QTextCursor(edit.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)

    def shortcutEscape(self):
        """ESC键功能"""
        if self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        elif self.canvas.editing_textblkitem is not None and self.canvas.editing_textblkitem.isEditing():
            self.canvas.editing_textblkitem.endEdit()

    # ====================== 模式切换方法 ======================
    def setPaintMode(self):
        """设置绘图模式"""
        if self.bottomBar.paintChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(0)
            self.canvas.setPaintMode(True)
            self.bottomBar.originalSlider.show()
            self.bottomBar.textlayerSlider.show()
            self.bottomBar.textblockChecker.hide()
        else:
            self.canvas.setPaintMode(False)
            self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)

    def setTextEditMode(self):
        """设置文本编辑模式"""
        if self.bottomBar.texteditChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.bottomBar.textblockChecker.show()
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.st_manager.setTextEditMode(True)
            self.setTextBlockMode()
        else:
            self.bottomBar.textblockChecker.hide()
            self.rightComicTransStackPanel.setHidden(True)
            self.st_manager.setTextEditMode(False)
        self.canvas.setPaintMode(False)

    def setTextBlockMode(self):
        """设置文本块模式"""
        mode = self.bottomBar.textblockChecker.isChecked()
        self.canvas.setTextBlockMode(mode)
        pcfg.imgtrans_textblock = mode
        self.st_manager.showTextblkItemRect(mode)

    # ====================== 项目保存方法 ======================
    def manual_save(self):
        """手动保存项目"""
        if self.leftBar.imgTransChecker.isChecked()\
            and self.imgtrans_proj.directory is not None:
            LOGGER.debug('Manually saving...')
            self.saveCurrentPage(update_scene_text=True, save_proj=True, restore_interface=True, save_rst_only=False)

    def saveCurrentPage(self, update_scene_text=True, save_proj=True, restore_interface=False, save_rst_only=False):
        """保存当前页面"""
        if not self.imgtrans_proj.img_valid:
            return
        
        # 保存前备份界面状态
        if restore_interface:
            set_canvas_focus = self.canvas.hasFocus()
            sel_textitem = self.canvas.selected_text_items()
            n_sel_textitems = len(sel_textitem)
            editing_textitem = None
            if n_sel_textitems == 1 and sel_textitem[0].isEditing():
                editing_textitem = sel_textitem[0]
        
        # 更新场景文本
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        
        # 确保在文本编辑模式
        if self.rightComicTransStackPanel.isHidden():
            self.bottomBar.texteditChecker.click()

        # 临时禁用文本块模式
        restore_textblock_mode = False
        if pcfg.imgtrans_textblock:
            restore_textblock_mode = True
            self.bottomBar.textblockChecker.click()

        # 隐藏文本形状控件
        hide_tsc = False
        if self.st_manager.txtblkShapeControl.isVisible():
            hide_tsc = True
            self.st_manager.txtblkShapeControl.hide()

        # 创建结果目录
        if not osp.exists(self.imgtrans_proj.result_dir()):
            os.makedirs(self.imgtrans_proj.result_dir())

        # 保存项目文件
        if save_proj:
            self.imgtrans_proj.save()
            if not save_rst_only:
                mask_path = self.imgtrans_proj.get_mask_path()
                mask_array = self.imgtrans_proj.mask_array
                self.imsave_thread.saveImg(mask_path, mask_array)
                inpainted_path = self.imgtrans_proj.get_inpainted_path()
                if self.canvas.drawingLayer.drawed():
                    inpainted = self.canvas.base_pixmap.copy()
                    painter = QPainter(inpainted)
                    painter.drawPixmap(0, 0, self.canvas.drawingLayer.get_drawed_pixmap())
                    painter.end()
                else:
                    inpainted = self.imgtrans_proj.inpainted_array
                self.imsave_thread.saveImg(inpainted_path, inpainted)

        # 渲染并保存结果图像
        img = self.canvas.render_result_img()
        imsave_path = self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img)
        self.imsave_thread.saveImg(imsave_path, img, self.imgtrans_proj.current_img, save_params={'ext': pcfg.imgsave_ext, 'quality': pcfg.imgsave_quality})
            
        # 更新保存状态
        self.canvas.setProjSaveState(False)
        self.canvas.update_saved_undostep()

        # 恢复界面状态
        if restore_interface:
            if restore_textblock_mode:
                self.bottomBar.textblockChecker.click()
            if hide_tsc:
                self.st_manager.txtblkShapeControl.show()
            if set_canvas_focus:
                self.canvas.setFocus()
            if n_sel_textitems > 0:
                self.canvas.block_selection_signal = True
                for blk in sel_textitem:
                    blk.setSelected(True)
                self.st_manager.on_incanvas_selection_changed()
                self.canvas.block_selection_signal = False
            if editing_textitem is not None:
                editing_textitem.startEdit()
        
    # ====================== 配置跳转方法 ======================
    def to_trans_config(self):
        """跳转到翻译配置"""
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnTranslator()

    def to_inpaint_config(self):
        """跳转到修复配置"""
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnInpaint()

    def to_ocr_config(self):
        """跳转到OCR配置"""
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnOCR()

    def to_detect_config(self):
        """跳转到文本检测配置"""
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnDetect()

    # ====================== 模块选择变化处理 ======================
    def on_textdet_changed(self):
        """文本检测器变化"""
        module = self.bottomBar.textdet_selector.selector.currentText()
        tgt_selector = self.configPanel.detect_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_TEXTDETECTORS():
            tgt_selector.setCurrentText(module)

    def on_ocr_changed(self):
        """OCR模块变化"""
        module = self.bottomBar.ocr_selector.selector.currentText()
        tgt_selector = self.configPanel.ocr_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_OCR():
            tgt_selector.setCurrentText(module)

    def on_trans_changed(self):
        """翻译器变化"""
        module = self.bottomBar.trans_selector.selector.currentText()
        tgt_selector = self.configPanel.trans_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_TRANSLATORS():
            tgt_selector.setCurrentText(module)

    def on_trans_src_changed(self):
        """翻译源语言变化"""
        sender = self.sender()
        text = sender.currentText()
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_source(text)
        pcfg.module.translate_source = text
        combobox = self.configPanel.trans_config_panel.source_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.src_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)

    def on_trans_tgt_changed(self):
        """翻译目标语言变化"""
        sender = self.sender()
        text = sender.currentText()
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_target(text)
        pcfg.module.translate_target = text
        combobox = self.configPanel.trans_config_panel.target_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.tgt_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)

    def on_inpaint_changed(self):
        """修复模块变化"""
        module = self.bottomBar.inpaint_selector.selector.currentText()
        tgt_selector = self.configPanel.inpaint_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_INPAINTERS():
            tgt_selector.setCurrentText(module)

    # ====================== 翻译功能方法 ======================
    def on_transpagebtn_pressed(self, run_target: bool):
        """翻译当前页面按钮按下"""
        page_key = self.imgtrans_proj.current_img
        if page_key is None:
            return

        blkitem_list = self.st_manager.textblk_item_list

        if len(blkitem_list) < 1:
            return
        
        self.translateBlkitemList(blkitem_list, -1)

    def translateBlkitemList(self, blkitem_list: List, mode: int) -> bool:
        """翻译文本块列表"""
        tgt_img = self.imgtrans_proj.img_array
        if tgt_img is None:
            return False
        tgt_mask = self.imgtrans_proj.mask_array
        
        if len(blkitem_list) < 1:
            return False
        
        self.global_search_widget.set_document_edited()
        
        im_h, im_w = tgt_img.shape[:2]

        blk_list, blk_ids = [], []
        for blkitem in blkitem_list:
            blk: TextBlock = blkitem.blk
            blk._bounding_rect = blkitem.absBoundingRect()
            blk.text = self.st_manager.pairwidget_list[blkitem.idx].e_source.toPlainText()
            blk_ids.append(blkitem.idx)
            blk.set_lines_by_xywh(blk._bounding_rect, angle=-blk.angle, x_range=[0, im_w-1], y_range=[0, im_h-1], adjust_bbox=True)
            blk_list.append(blk)

        self.module_manager.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask = tgt_mask)
        return True

    def finishTranslatePage(self, page_key):
        """完成页面翻译"""
        if page_key == self.imgtrans_proj.current_img:
            self.st_manager.updateTranslation()

    def on_imgtrans_pipeline_finished(self):
        """图像翻译流程完成"""
        self.backup_blkstyles.clear()
        self._run_imgtrans_wo_textstyle_update = False
        self.postprocess_mt_toggle = True
        if pcfg.module.empty_runcache and not shared.HEADLESS:
            self.module_manager.unload_all_models()
        if shared.args.export_translation_txt:
            self.on_export_txt('translation')
        if shared.args.export_source_txt:
            self.on_export_txt('source')
        if shared.HEADLESS:
            self.run_next_dir()

    def postprocess_translations(self, blk_list: List[TextBlock]) -> None:
        """翻译后处理"""
        src_is_cjk = is_cjk(pcfg.module.translate_source)
        tgt_is_cjk = is_cjk(pcfg.module.translate_target)
        if tgt_is_cjk:
            for blk in blk_list:
                if src_is_cjk:
                    blk.translation = full_len(blk.translation)
                else:
                    blk.translation = half_len(blk.translation)
                    blk.translation = re.sub(r'([?.!"])\s+', r'\1', blk.translation)    # 移除标点后的空格
        else:
            for blk in blk_list:
                if blk.vertical:
                    blk.alignment = TextAlignment.Center
                blk.translation = half_len(blk.translation)
                blk.vertical = False

        # 应用关键词替换
        for blk in blk_list:
            blk.translation = self.mtSubWidget.sub_text(blk.translation)
            if pcfg.let_uppercase_flag:
                blk.translation = blk.translation.upper()

    def on_pagtrans_finished(self, page_index: int):
        """页面翻译完成"""
        blk_list = self.imgtrans_proj.get_blklist_byidx(page_index)
        ffmt_list = None
        if len(self.backup_blkstyles) == self.imgtrans_proj.num_pages and len(self.backup_blkstyles[page_index]) == len(blk_list):
            ffmt_list: List[FontFormat] = self.backup_blkstyles[page_index]

        self.postprocess_translations(blk_list)
                
        # 应用全局字体格式
        override_fnt_size = pcfg.let_fntsize_flag == 1
        override_fnt_stroke = pcfg.let_fntstroke_flag == 1
        override_fnt_color = pcfg.let_fntcolor_flag == 1
        override_fnt_scolor = pcfg.let_fnt_scolor_flag == 1
        override_alignment = pcfg.let_alignment_flag == 1
        override_effect = pcfg.let_fnteffect_flag == 1
        override_writing_mode = pcfg.let_writing_mode_flag == 1
        override_font_family = pcfg.let_family_flag == 1
        gf = self.textPanel.formatpanel.global_format

        inpaint_only = pcfg.module.enable_inpaint
        inpaint_only = inpaint_only and not (pcfg.module.enable_detect or pcfg.module.enable_ocr or pcfg.module.enable_translate)
        
        if not inpaint_only:
            for ii, blk in enumerate(blk_list):
                if self._run_imgtrans_wo_textstyle_update and ffmt_list is not None:
                    blk.fontformat.merge(ffmt_list[ii])
                else:
                    # 应用全局格式覆盖
                    if override_fnt_size or blk.font_size < 0:
                        blk.font_size = gf.font_size
                    elif blk._detected_font_size > 0 and not pcfg.module.enable_detect:
                        blk.font_size = blk._detected_font_size
                    if override_fnt_stroke:
                        blk.stroke_width = gf.stroke_width
                    elif pcfg.module.enable_ocr:
                        blk.recalulate_stroke_width()
                    if override_fnt_color:
                        blk.set_font_colors(fg_colors=gf.frgb)
                    if override_fnt_scolor:
                        blk.set_font_colors(bg_colors=gf.srgb)
                    if override_alignment:
                        blk.alignment = gf.alignment
                    elif pcfg.module.enable_detect and not blk.src_is_vertical:
                        blk.recalulate_alignment()
                    if override_effect:
                        blk.opacity = gf.opacity
                        blk.shadow_color = gf.shadow_color
                        blk.shadow_radius = gf.shadow_radius
                        blk.shadow_strength = gf.shadow_strength
                        blk.shadow_offset = gf.shadow_offset
                    if override_writing_mode:
                        blk.vertical = gf.vertical
                    if override_font_family or blk.font_family is None:
                        blk.font_family = gf.font_family
                        if blk.rich_text:
                            blk.rich_text = set_html_family(blk.rich_text, gf.font_family)
                    
                    # 应用其他格式属性
                    blk.line_spacing = gf.line_spacing
                    blk.letter_spacing = gf.letter_spacing
                    blk.italic = gf.italic
                    blk.bold = gf.bold
                    blk.underline = gf.underline
                    sw = blk.stroke_width
                    if sw > 0 and pcfg.module.enable_ocr and pcfg.module.enable_detect and not override_fnt_size:
                        blk.font_size = blk.font_size / (1 + sw)

            self.st_manager.auto_textlayout_flag = pcfg.let_autolayout_flag and \
                (pcfg.module.enable_detect or pcfg.module.enable_translate)
        
        # 更新当前页面
        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.imgtrans_proj.set_current_img_byidx(page_index)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()

        # 特殊处理：仅启用翻译时调整文本块
        if not pcfg.module.enable_detect and pcfg.module.enable_translate:
            for blkitem in self.st_manager.textblk_item_list:
                blkitem.squeezeBoundingRect()

        if page_index + 1 == self.imgtrans_proj.num_pages:
            self.st_manager.auto_textlayout_flag = False

        # 保存项目文件
        self.imgtrans_proj.save()

        # 保存当前页面
        self.saveCurrentPage(False, False)

    # ====================== 状态变化处理 ======================
    def on_savestate_changed(self, unsaved: bool):
        """保存状态变化"""
        save_state = self.tr('unsaved') if unsaved else self.tr('saved')
        self.titleBar.setTitleContent(save_state=save_state)

    def on_textstack_changed(self):
        """文本堆栈变化"""
        if not self.page_changing:
            self.global_search_widget.set_document_edited()

    def on_run_blktrans(self, mode: int):
        """运行块翻译"""
        blkitem_list = self.canvas.selected_text_items()
        self.translateBlkitemList(blkitem_list, mode)

    def on_blktrans_finished(self, mode: int, blk_ids: List[int]):
        """块翻译完成"""
        if len(blk_ids) < 1:
            return
        
        blkitem_list = [self.st_manager.textblk_item_list[idx] for idx in blk_ids]
        pairw_list = []
        for blk in blkitem_list:
            pairw_list.append(self.st_manager.pairwidget_list[blk.idx])
        self.canvas.push_undo_command(RunBlkTransCommand(self.canvas, blkitem_list, pairw_list, mode))

    def on_imgtrans_progressbox_showed(self):
        """翻译进度框显示"""
        msg_size = self.module_manager.progress_msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        self.module_manager.progress_msgbox.move(p)

    # ====================== 窗口事件处理 ======================
    def on_closebtn_clicked(self):
        """关闭按钮点击事件"""
        if self.imsave_thread.isRunning():
            self.imsave_thread.finished.connect(self.close)
            mb = FrameLessMessageBox()
            mb.setText(self.tr('Saving image...'))
            self.imsave_thread.finished.connect(mb.close)
            mb.exec()
            return
        self.close()

    def on_display_lang_changed(self, lang: str):
        """显示语言变化"""
        if lang != pcfg.display_lang:
            pcfg.display_lang = lang
            self.set_display_lang(lang)

    # ====================== 翻译运行控制 ======================
    def run_imgtrans(self):
        """运行图像翻译"""
        if not self.imgtrans_proj.is_all_pages_no_text and not pcfg.module.keep_exist_textlines:
            reply = QMessageBox.question(self, self.tr('Confirmation'),
                                         self.tr('Are you sure to run image translation again?\nAll existing translation results will be cleared!'),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.on_run_imgtrans()

    def run_imgtrans_wo_textstyle_update(self):
        """不更新文本样式运行翻译"""
        self._run_imgtrans_wo_textstyle_update = True
        self.run_imgtrans()

    def on_run_imgtrans(self):
        """执行图像翻译"""
        self.backup_blkstyles.clear()

        # 确保不在文本块模式
        if self.bottomBar.textblockChecker.isChecked():
            self.bottomBar.textblockChecker.click()
        self.postprocess_mt_toggle = False

        # 清除现有翻译结果（如果配置要求）
        all_disabled = pcfg.module.all_stages_disabled()
        if pcfg.module.enable_detect:
            for page in self.imgtrans_proj.pages:
                if not pcfg.module.keep_exist_textlines:
                    self.imgtrans_proj.pages[page].clear()
        else:
            self.st_manager.updateTextBlkList()
            textblk: TextBlock = None
            for blklist in self.imgtrans_proj.pages.values():
                ffmt_list = []
                self.backup_blkstyles.append(ffmt_list)
                for textblk in blklist:
                    if not pcfg.module.enable_detect:
                        ffmt_list.append(textblk.fontformat.deepcopy())
                    if pcfg.module.enable_ocr:
                        textblk.text = []
                        textblk.set_font_colors((0, 0, 0), (0, 0, 0))
                    if pcfg.module.enable_translate or (all_disabled and not self._run_imgtrans_wo_textstyle_update) or pcfg.module.enable_ocr:
                        textblk.rich_text = ''
                    textblk.vertical = textblk.src_is_vertical
        # 运行翻译流程
        self.module_manager.runImgtransPipeline()

    # ====================== 面板变化处理 ======================
    def on_transpanel_changed(self):
        """翻译面板变化"""
        self.canvas.editor_index = self.rightComicTransStackPanel.currentIndex()
        if not self.canvas.textEditMode() and self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        self.canvas.updateLayers()

    # ====================== 文本样式管理 ======================
    def import_tstyles(self):
        """导入文本样式"""
        ddir = osp.dirname(pcfg.text_styles_path)
        p = QFileDialog.getOpenFileName(self, self.tr("Import Text Styles"), ddir, None, "(.json)")
        if not isinstance(p, str):
            p = p[0]
        if p == '':
            return
        try:
            load_textstyle_from(p, raise_exception=True)
            save_config()
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed to load from {p}'))

    def export_tstyles(self):
        """导出文本样式"""
        ddir = osp.dirname(pcfg.text_styles_path)
        savep = QFileDialog.getSaveFileName(self, self.tr("Save Text Styles"), ddir, None, "(.json)")
        if not isinstance(savep, str):
            savep = savep[0]
        if savep == '':
            return
        suffix = Path(savep).suffix
        if suffix != '.json':
            if suffix == '':
                savep = savep + '.json'
            else:
                savep = savep.replace(suffix, '.json')
        oldp = pcfg.text_styles_path
        try:
            pcfg.text_styles_path = savep
            save_text_styles(raise_exception=True)
            save_config()
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed save to {savep}'))
            pcfg.text_styles_path = oldp

    # ====================== 文本区域控制 ======================
    def fold_textarea(self, fold: bool):
        """折叠/展开文本区域"""
        pcfg.fold_textarea = fold
        self.textPanel.textEditList.setFoldTextarea(fold)

    def show_source_text(self, show: bool):
        """显示源文本"""
        pcfg.show_source_text = show
        self.textPanel.textEditList.setSourceVisible(show)

    def show_trans_text(self, show: bool):
        """显示翻译文本"""
        pcfg.show_trans_text = show
        self.textPanel.textEditList.setTransVisible(show)

    # ====================== 文档导入导出 ======================
    def on_export_doc(self):
        """导出文档"""
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()
        self.export_doc_thread.exportAsDoc(self.imgtrans_proj)

    def on_import_doc(self):
        """导入文档"""
        self.import_doc_thread.importDoc(self.imgtrans_proj)

    def on_export_txt(self, dump_target, suffix='.txt'):
        """导出文本文件"""
        try:
            self.imgtrans_proj.dump_txt(dump_target=dump_target, suffix=suffix)
            create_info_dialog(self.tr('Text file exported to ') + self.imgtrans_proj.dump_txt_path(dump_target, suffix))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export as TEXT file'))

    def on_import_trans_txt(self):
        """导入翻译文本"""
        try:
            selected_file = ''
            dialog = QFileDialog()
            selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import *.md/*.txt'), filter="*.txt *.md *.TXT *.MD")[0].toLocalFile())
            if not osp.exists(selected_file):
                return

            # 加载翻译文本
            all_matched, match_rst = self.imgtrans_proj.load_translation_from_txt(selected_file)
            matched_pages = match_rst['matched_pages']

            # 更新当前页面
            if self.imgtrans_proj.current_img in matched_pages:
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()

            # 显示匹配结果
            if all_matched:
                msg = self.tr('Translation imported and matched successfully.')
            else:
                msg = self.tr('Imported txt file not fully matched with current project, please make sure source txt file structured like results from \"export TXT/markdown\"')
                if len(match_rst['missing_pages']) > 0:
                    msg += '\n' + self.tr('Missing pages: ') + '\n'
                    msg += '\n'.join(match_rst['missing_pages'])
                if len(match_rst['unexpected_pages']) > 0:
                    msg += '\n' + self.tr('Unexpected pages: ') + '\n'
                    msg += '\n'.join(match_rst['unexpected_pages'])
                if len(match_rst['unmatched_pages']) > 0:
                    msg += '\n' + self.tr('Unmatched pages: ') + '\n'
                    msg += '\n'.join(match_rst['unmatched_pages'])
                msg = msg.strip()
            create_info_dialog(msg)

        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import translation from ') + selected_file)

    # ====================== 文件操作 ======================
    def on_reveal_file(self):
        """在文件资源管理器中显示文件"""
        current_img_path = self.imgtrans_proj.current_img_path()
        if sys.platform == 'win32':
            p = "\""+str(Path(current_img_path))+"\""
            subprocess.Popen("explorer.exe /select,"+p, shell=True)
        elif sys.platform == 'darwin':
            p = "\""+current_img_path+"\""
            subprocess.Popen("open -R "+p, shell=True)

    # ====================== 全局搜索控制 ======================
    def on_set_gsearch_widget(self):
        """设置全局搜索组件"""
        setup = self.leftBar.globalSearchChecker.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            self.leftBar.showPageListLabel.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.global_search_widget)
        else:
            self.leftStackWidget.hide()

    # ====================== 导入导出完成处理 ======================
    def on_fin_export_doc(self):
        """文档导出完成"""
        msg = QMessageBox()
        msg.setText(self.tr('Export to ') + self.imgtrans_proj.doc_path())
        msg.exec_()

    def on_fin_import_doc(self):
        """文档导入完成"""
        self.st_manager.updateSceneTextitems()

    # ====================== 全局替换完成处理 ======================
    def on_global_replace_finished(self):
        """全局替换完成"""
        rt = self.global_search_widget.replace_thread
        self.canvas.push_text_command(
            GlobalRepalceAllCommand(rt.sceneitem_list, rt.background_list, rt.target_text, self.imgtrans_proj)
        )
        rt.sceneitem_list = None
        rt.background_list = None

    # ====================== 暗黑模式切换 ======================
    def on_darkmode_triggered(self):
        """暗黑模式切换"""
        pcfg.darkmode = self.titleBar.darkModeAction.isChecked()
        self.resetStyleSheet(reverse_icon=True)
        self.save_config()

    # ====================== 文本处理钩子 ======================
    def ocr_postprocess(self, textblocks: List[TextBlock], img, ocr_module=None, **kwargs):
        """OCR后处理"""
        for blk in textblocks:
            text = blk.get_text()
            blk.text = self.ocrSubWidget.sub_text(text)

    def translate_preprocess(self, translations: List[str] = None, textblocks: List[TextBlock] = None, translator = None, source_text:list = []):
        """翻译预处理"""
        for i in range(len(source_text)):
            source_text[i] = self.mtPreSubWidget.sub_text(source_text[i])

    def translate_postprocess(self, translations: List[str] = None, textblocks: List[TextBlock] = None, translator = None):
        """翻译后处理"""
        if not self.postprocess_mt_toggle:
            return
        
        for ii, tr in enumerate(translations):
            translations[ii] = self.mtSubWidget.sub_text(tr)

    # ====================== 文本复制粘贴 ======================
    def on_copy_src(self):
        """复制源文本"""
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        
        # 针对GPT翻译器特殊处理
        if isinstance(self.module_manager.translator, GPTTranslator):
            src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText() for blk in blks]
            src_txt = ''
            for (prompt, num_src) in self.module_manager.translator._assemble_prompts(src_list, max_tokens=4294967295):
                src_txt += prompt
            src_txt = src_txt.strip()
        else:
            src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText().strip().replace('\n', ' ') for blk in blks]
            src_txt = '\n'.join(src_list)

        self.st_manager.app_clipborad.setText(src_txt, QClipboard.Mode.Clipboard)

    def on_paste_src(self):
        """粘贴源文本"""
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return

        src_widget_list = [self.st_manager.pairwidget_list[blk.idx].e_source for blk in blks]
        text_list = self.st_manager.app_clipborad.text().split('\n')
        
        n_paragraph = min(len(src_widget_list), len(text_list))
        if n_paragraph < 1:
            return
        
        src_widget_list = src_widget_list[:n_paragraph]
        text_list = text_list[:n_paragraph]

        self.canvas.push_undo_command(PasteSrcItemsCommand(src_widget_list, text_list))

    # ====================== 键盘事件处理 ======================
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """按键事件"""
        key = event.key()
        if hasattr(self, 'canvas'):
            if key == Qt.Key.Key_Alt:
                self.canvas.alt_pressed = True
        return super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """按键释放事件"""
        if hasattr(self, 'canvas'):
            if event.key() == Qt.Key.Key_Alt:
                self.canvas.alt_pressed = False
                if self.canvas.scale_tool_mode:
                    self.canvas.scale_tool_mode = False
                    self.canvas.end_scale_tool.emit()
        return super().keyReleaseEvent(event)
    
    # ====================== 批处理功能 ======================
    def run_batch(self, exec_dirs: Union[List, str], **kwargs):
        """运行批处理"""
        if not isinstance(exec_dirs, List):
            exec_dirs = exec_dirs.split(',')
        valid_dirs = []
        for d in exec_dirs:
            if osp.exists(d):
                valid_dirs.append(d)
            else:
                LOGGER.warning(f'target directory {d} does not exist.')
        self.exec_dirs = valid_dirs
        self.run_next_dir()

    def run_next_dir(self):
        """处理下一个目录"""
        if len(self.exec_dirs) == 0:
            while self.imsave_thread.isRunning():
                time.sleep(0.1)
            LOGGER.info(f'finished translating all dirs, quit app...')
            self.app.quit()
            return
        d = self.exec_dirs.pop(0)
        
        LOGGER.info(f'translating {d} ...')
        self.openDir(d)
        shared.pbar = {}
        npages = len(self.imgtrans_proj.pages)
        if npages > 0:
            # 初始化进度条
            if pcfg.module.enable_detect:
                shared.pbar['detect'] = tqdm(range(npages), desc="Text Detection")
            if pcfg.module.enable_ocr:
                shared.pbar['ocr'] = tqdm(range(npages), desc="OCR")
            if pcfg.module.enable_translate:
                shared.pbar['translate'] = tqdm(range(npages), desc="Translation")
            if pcfg.module.enable_inpaint:
                shared.pbar['inpaint'] = tqdm(range(npages), desc="Inpaint")
        self.on_run_imgtrans()

    # ====================== 对话框创建 ======================
    def on_create_errdialog(self, error_msg: str, detail_traceback: str = '', exception_type: str = ''):
        """创建错误对话框"""
        try:
            if exception_type != '':
                shared.showed_exception.add(exception_type)
            err = QMessageBox()
            err.setText(error_msg)
            err.setDetailedText(detail_traceback)
            err.exec()
            if exception_type != '':
                shared.showed_exception.remove(exception_type)
        except:
            if exception_type in shared.showed_exception:
                shared.showed_exception.remove(exception_type)
            LOGGER.error('Failed to create error dialog')
            LOGGER.error(traceback.format_exc())

    def on_create_infodialog(self, info_dict: dict):
        """创建信息对话框"""
        QMessageBox.StandardButton.NoButton
        dialog = MessageBox(**info_dict)
        dialog.show()   # 非阻塞显示

    # ====================== 视图组件注册 ======================
    def setupRegisterWidget(self):
        """注册视图组件"""
        self.titleBar.viewMenu.addSeparator()
        for cfg_name in shared.config_name_to_view_widget:
            d = shared.config_name_to_view_widget[cfg_name]
            widget: ViewWidget = d['widget']
            action = QAction(widget.action_name, self.titleBar)
            action.setCheckable(True)
            visible = getattr(pcfg, cfg_name)
            action.setChecked(visible)
            action.triggered.connect(self.action_set_view_visible)
            self.titleBar.viewMenu.addAction(action)
            d['action'] = action
            shared.action_to_view_config_name[action] = cfg_name
            widget.set_expend_area(expend=getattr(pcfg, widget.config_expand_name), set_config=False)
            widget.view_hide_btn_clicked.connect(self.on_hide_view_widget)
            widget.setVisible(visible)

    def register_view_widget(self, widget: ViewWidget):
        """注册视图组件"""
        assert widget.config_name not in shared.config_name_to_view_widget
        d = {'widget': widget}
        shared.config_name_to_view_widget[widget.config_name] = d

    def action_set_view_visible(self):
        """设置视图可见性"""
        action: QAction = self.sender()
        show = action.isChecked()
        cfg_name = shared.action_to_view_config_name[action]
        widget: ViewWidget = shared.config_name_to_view_widget[cfg_name]['widget']
        widget.setVisible(show)
        setattr(pcfg, cfg_name, show)

    def on_hide_view_widget(self, cfg_name: str):
        """隐藏视图组件"""
        d = shared.config_name_to_view_widget[cfg_name]
        widget: ViewWidget = d['widget']
        widget.setVisible(False)
        action: QAction = d['action']
        action.setChecked(False)
        setattr(pcfg, cfg_name, False)