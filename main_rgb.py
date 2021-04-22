"""
Copyright {2018} {Viraj Mavani}
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
"""
import traceback
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import cv2

# make sure the file is inside semi-auto-image-annotation-tool-master
import pathlib

import config

from templateMatch import template_match
from csv_edit import *
from label_file import label_names

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

cur_path = pathlib.Path(__file__).parent.absolute().as_posix()
sys.path.append(cur_path)
os.chdir(cur_path)


class MainGUI:
    def __init__(self, master, root_w, root_h):

        self.detect_label = False
        # to choose between keras or tensorflow models
        self.keras_ = 1  # default
        self.tensorflow_ = 0
        self.models_dir = ''  # gets updated as per user choice
        self.model_path = ''
        self.parent = master
        self.parent.title("Semi Automatic Image Annotation Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=False, height=False)

        # Initialize class variables
        self.video = None
        self.img = None
        self.tkimg = None
        self.imageDir = ''
        self.imageDirPathBuffer = ''
        self.imageList = []
        self.imageTotal = 0
        self.imageCur = 0
        self.cur = 0
        self.bboxIdList = []
        self.ovalIdList = []
        self.bboxOvalDict = {}
        self.bboxList = []
        self.bboxPointList = []
        self.templateList = []
        self.depth_templateList = []
        self.o1 = None
        self.o2 = None
        self.o3 = None
        self.o4 = None
        self.bboxId = None
        self.currLabel = None
        self.editbboxId = None
        self.currBboxColor = None
        self.zoomImgId = None
        self.zoomImg = None
        self.zoomImgCrop = None
        self.tkZoomImg = None
        self.hl = None
        self.vl = None
        self.editPointId = None
        self.filename = None
        self.filenameBuffer = None
        self.objectLabelList = []
        self.EDIT = False
        self.autoSuggest = StringVar()
        self.writer = None
        self.initHelpText = None
        self.org_h = 0
        self.org_w = 0

        # 显示窗口size
        self.win_h = 720
        self.win_w = 1280

        # 图像缩放比例
        self.rate = 0

        # initialize mouse state
        self.STATE = {'x': 0, 'y': 0}
        self.STATE_COCO = {'click': 0}

        self.click = False

        # initialize annotation file
        self.anno_filename = 'annotations.csv'
        self.annotation_file = open('annotations/' + self.anno_filename, 'w+')
        self.annotation_file.write("")
        self.annotation_file.close()

        # ------------------ GUI ---------------------

        # Control Panel
        self.ctrlPanel = Frame(self.frame, height=root_h)
        self.ctrlPanel.grid(row=0, column=0, sticky=W + N)
        self.openBtn = Button(self.ctrlPanel, text='Open', command=self.open_image)
        self.openBtn.grid(columnspan=2, sticky=W + E)
        self.openDirBtn = Button(self.ctrlPanel, text='Open Dir', command=self.open_image_dir)
        self.openDirBtn.grid(columnspan=2, sticky=W + E)

        self.nextBtn = Button(self.ctrlPanel, text='Next -->', command=self.open_next)
        self.nextBtn.grid(columnspan=2, sticky=W + E)
        self.previousBtn = Button(self.ctrlPanel, text='<-- Previous', command=self.open_previous)
        self.previousBtn.grid(columnspan=2, sticky=W + E)

        self.stepBox = Entry(self.ctrlPanel, text="Enter frame number")
        self.stepBox.grid(columnspan=2, sticky=W + E)
        self.stepBtn = Button(self.ctrlPanel, text="Step To", command=self.step_to)
        self.stepBtn.grid(columnspan=2, sticky=W + E)

        self.saveBtn = Button(self.ctrlPanel, text='Save', command=self.save)
        self.saveBtn.grid(columnspan=2, sticky=W + E)

        self.semiAutoBtn = Button(self.ctrlPanel, text="Detect", command=self.automate)
        self.semiAutoBtn.grid(columnspan=2, sticky=W + E)

        self.brightnessBox = Entry(self.ctrlPanel, text="Enter brightness enhance factor")
        self.brightnessBox.grid(columnspan=2, sticky=W + E)

        self.disp = Label(self.ctrlPanel, text='Coordinates:')
        self.disp.grid(columnspan=2, sticky=W + E)

        self.initHelpText = Listbox(self.ctrlPanel, exportselection=False)
        self.initHelpText.grid(columnspan=2, sticky=W + E)

        self.zoomPanelLabel = Label(self.ctrlPanel, text="Precision View Panel")
        self.zoomPanelLabel.grid(columnspan=2, sticky=W + E)
        self.zoomcanvas = Canvas(self.ctrlPanel, width=300, height=300)
        self.zoomcanvas.grid(columnspan=2, sticky=W + E)

        # Image Editing Region
        self.canvas = Canvas(self.frame, width=self.win_w, height=self.win_h)
        self.canvas.grid(row=0, column=1, sticky=W + E)
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.bind("<Button-3>", self.right_click)
        self.canvas.bind("<Motion>", self.mouse_move, "+")
        self.parent.bind("<Key-Left>", self.open_previous)
        self.parent.bind("<Key-Right>", self.open_next)
        self.parent.bind("<Key>", self.key_control)

        # self.depth_canvas = Canvas(self.frame, width=1000, height=480)
        # self.depth_canvas.grid(row=1, column=1, sticky=W + E)

        # Labels and Bounding Box Lists Panel
        self.listPanel = Frame(self.frame)
        self.listPanel.grid(row=0, column=2, sticky=NE)
        self.listBoxNameLabel = Label(self.listPanel, text="List of Objects").pack(fill=X, side=TOP)
        self.objectListBox = Listbox(self.listPanel, width=40, exportselection=False)
        self.objectListBox.pack(fill=X, side=TOP)
        self.objectListBox.bind("<<ListboxSelect>>", self.label_select)

        self.labelBox = Entry(self.listPanel, text="Enter label")
        self.labelBox.pack(fill=X, side=TOP)
        self.changeLabelBtn = Button(self.listPanel, text="Change Label", command=self.change_label)
        self.changeLabelBtn.pack(fill=X, side=TOP)

        self.delObjectBtn = Button(self.listPanel, text="Delete", command=self.del_bbox)
        self.delObjectBtn.pack(fill=X, side=TOP)
        self.clearAllBtn = Button(self.listPanel, text="Clear All", command=self.clear_bbox)
        self.clearAllBtn.pack(fill=X, side=TOP)

        self.labelListBox = Listbox(self.listPanel, exportselection=False)
        self.labelListBox.pack(fill=X, side=TOP)

        self.fileListBox = Listbox(self.listPanel, exportselection=False)
        self.fileListBox.pack(fill=X, side=TOP)
        self.fileListBox.bind("<<ListboxSelect>>", self.file_select)

        self.cocoLabels = label_names

        self.cocoIntVars = []
        # print(self.cocoIntVars)

        self.modelIntVars = []

        # STATUS BAR
        self.statusBar = Frame(self.frame, width=500)
        self.statusBar.grid(row=1, column=1, sticky=W + N)
        self.processingLabel = Label(self.statusBar, text="                      ")
        self.processingLabel.pack(side="left", fill=X)
        self.imageIdxLabel = Label(self.statusBar, text="                      ")
        self.imageIdxLabel.pack(side="right", fill=X)

        self.add_all_classes()

    def open_image(self):
        self.filename = filedialog.askopenfilename(title="Select Image", filetypes=(("MP4", "*.mp4"),
                                                                                    ("all files", "*.*")))
        if not self.filename:
            return None
        self.filenameBuffer = self.filename
        self.load_image(self.filenameBuffer)

    def open_image_dir(self):
        self.imageDir = filedialog.askdirectory(title="Select Dataset Directory")
        if not self.imageDir:
            return None
        self.imageList = os.listdir(self.imageDir)
        self.imageList = [item for item in self.imageList if (os.path.splitext(item)[1] == '.jpg')]

        self.imageList = sorted(self.imageList)

        # self.add_file_list()

        self.imageTotal = len(self.imageList)
        self.filename = None
        self.cur = 0
        self.imageDirPathBuffer = self.imageDir
        self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])

        file_name = 'annotations'
        if len(self.imageDirPathBuffer) > 0:
            file_name = os.path.split(self.imageDirPathBuffer)[-1]
        file_name = 'annotations/' + file_name
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        self.anno_filename = file_name
        self.clear_bbox()
        bboxes, label_list = self.get_previous_result()

        self.add_boundingbox(bboxes, label_list)

    def get_previous_result(self):
        bboxes, label_list = get_previous(self.anno_filename, self.imageList[self.cur])
        bboxes = self.resize_box(bboxes)
        return bboxes, label_list

    def add_file_list(self):
        for file in self.imageList:
            # if self.cocoIntVars[listidxcoco].get():
            curr_file_list = self.fileListBox.get(0, END)
            curr_file_list = list(curr_file_list)
            # file_tip = os.path.join(self.imageDir, file)
            if file not in curr_file_list:
                self.fileListBox.insert(END, str(file))

    def load_image(self, file):

        # img = cv2.imread(file)
        # # img = gama_correct(img)
        # self.img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.img = Image.open(file)

        factor = self.brightnessBox.get()
        if factor is not None and factor != '':
            factor = float(factor)
            enh_bri = ImageEnhance.Brightness(self.img)
            self.img = enh_bri.enhance(factor)

        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(
            text='{}  ||   Image Number: {} / {}'.format(os.path.split(file)[-1], self.imageCur, self.imageTotal))
        # Resize to Pascal VOC format
        w, h = self.img.size
        self.org_w, self.org_h = self.img.size

        self.rate = min(self.win_w / self.org_w, self.win_h / self.org_h)
        self.img = self.img.resize((int(self.org_w * self.rate), int(self.org_h * self.rate)))

        self.click = False

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)

    def resize_box(self, boxes):
        if len(boxes) == 0:
            return boxes
        return np.asarray(np.array(boxes) * self.rate, np.int)

    def de_resize_box(self, boxes):
        if len(boxes) == 0:
            return boxes
        return np.asarray(np.array(boxes) / self.rate, np.int)

    def step_to(self):
        frame = int(self.stepBox.get())
        self.step_to_frame(frame)

    def step_to_frame(self, frame):
        if frame >= 0 and frame < self.imageTotal:
            self.cur = frame
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
            bboxes, label_list = self.get_previous_result()
            self.add_boundingbox(bboxes, label_list)
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def open_next(self, event=None):
        self.save()
        if self.cur < self.imageTotal - 1:
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
            self.automate()
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def open_previous(self, event=None):
        if self.cur > 0:
            self.cur -= 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
            bboxes, label_list = self.get_previous_result()
            self.add_boundingbox(bboxes, label_list)

        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def open_next_not_automate(self, event=None):
        self.save()
        if self.cur < self.imageTotal - 1:
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
            bboxes, label_list = self.get_previous_result()
            self.add_boundingbox(bboxes, label_list)
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def key_control(self, event):
        if event.keysym == 'Delete':
            self.del_bbox()
        elif event.keysym == 'BackSpace':
            self.clear_bbox()
        elif event.char == '+':
            self.open_next_not_automate()
        elif event.char == 's':
            self.save()
        elif event.char in ['a', 'w', 'd', 'z', 'j', 'i', 'l', 'm']:
            sel = self.objectListBox.curselection()
            if len(sel) != 1:
                return
            idx = sel[0]
            bboxId = self.bboxIdList[idx]

            x1, y1, x2, y2 = self.bboxList[idx]
            if event.char == 'a':
                x1 -= 1
            elif event.char == 'd':
                x1 += 1
            elif event.char == 'w':
                y1 -= 1
            elif event.char == 'z':
                y1 += 1
            elif event.char == 'j':
                x2 -= 1
            elif event.char == 'l':
                x2 += 1
            elif event.char == 'i':
                y2 -= 1
            elif event.char == 'm':
                y2 += 1
            x1, y1, x2, y2 = self.limate_box([x1, y1, x2, y2])
            self.canvas.coords(bboxId, (x1, y1, x2, y2))
            self.bboxList.pop(idx)
            self.bboxList.insert(idx, (x1, y1, x2, y2))

            self.objectListBox.delete(idx)
            self.objectListBox.insert(idx, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + self.objectLabelList[idx])
            self.objectListBox.itemconfig(idx,
                                          fg=config.COLORS[idx % len(config.COLORS)])
            self.objectListBox.select_clear(0, len(self.bboxIdList) - 1)
            self.objectListBox.select_set(idx)

    def save(self):
        img = self.img.copy()
        self.templateList = []
        self.depth_templateList = []
        if len(self.bboxList) > 0:
            for idx, item in enumerate(self.bboxList):
                x1, y1, x2, y2 = self.bboxList[idx]
                self.templateList.append(img.crop((x1, y1, x2, y2)))
            self.saveProcess(self.anno_filename, self.imageList[self.cur], self.imageDirPathBuffer, self.bboxList
                             , self.objectLabelList)
        else:
            anno_file = self.anno_filename + '/' + os.path.splitext(self.imageList[self.cur])[0] + '.csv'
            if os.path.exists(anno_file):
                os.remove(anno_file)

    def saveProcess(self, anno_filename, image, imageDirPathBuffer, bboxList, objectLabelList):
        try:
            annotation_file = open(anno_filename + '/' + os.path.splitext(image)[0] + '.csv', 'w', encoding='utf-8')

            resize_boxes = self.de_resize_box(bboxList)
            for idx, item in enumerate(resize_boxes):
                annotation_file.write(
                    imageDirPathBuffer + '/' + image + ',' + ','.join(
                        map(str, item)) + ','
                    + str(objectLabelList[idx].split(':')[0]) + '\n')

            annotation_file.close()
        except:
            print('error')
            traceback.print_exc(file=sys.stdout)

    def mouse_click(self, event):

        ox1, oy1 = (event.x - 3), (event.y - 3)
        ox2, oy2 = (event.x + 3), (event.y + 3)

        oval_id = self.canvas.create_oval(ox1, oy1, ox2, oy2,
                                          fill=config.COLORS[len(self.bboxList) % len(config.COLORS)])
        self.ovalIdList.append(oval_id)

        # Check if Updating BBox
        if self.click:
            try:
                labelidx = self.labelListBox.curselection()
                self.currLabel = self.labelListBox.get(labelidx)
            except:
                pass
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            bbox_id = self.add_box([x1, y1, x2, y2], str(self.currLabel))
            self.bboxOvalDict[bbox_id] = [self.ovalIdList[-1], self.ovalIdList[-2]]
            self.currLabel = None

            if self.objectListBox.size() > 0:
                self.objectListBox.select_clear(0, len(self.bboxIdList) - 1)
                self.objectListBox.select_set(len(self.bboxIdList) - 1)
                self.label_select(None)
            self.click = False
        else:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
            self.click = True

    def right_click(self, event):
        if self.click:
            self.canvas.delete(self.ovalIdList[-1])
            self.click = False

    def label_select(self, event):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])

        for item_id in self.bboxIdList:
            self.canvas.itemconfig(item_id, width=2)
        self.canvas.itemconfig(self.bboxIdList[idx], width=4)

    def file_select(self, event):
        sel = self.fileListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.step_to_frame(idx)
        # self.fileListBox.select_clear(0, len(self.imageList) - 1)

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        self.zoom_view(event)

        if self.tkimg:
            # Horizontal and Vertical Line for precision
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y,
                                              width=2, fill='green', dash=(4, 4))
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(),
                                              width=2, fill='green', dash=(4, 4))

    def zoom_view(self, event):
        # try:
        if self.img is not None:

            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            crop_image = self.zoomImg.crop(((event.x - 30), (event.y - 30), (event.x + 30), (event.y + 30)))
            crop_image = crop_image.resize((300, 300))
            enh_bri = ImageEnhance.Brightness(crop_image)
            self.zoomImgCrop = enh_bri.enhance(factor=1.5)

            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 150, 300, 150, width=2)
            vl = self.zoomcanvas.create_line(150, 0, 150, 300, width=2)
        # except:
        #
        #     pass

    def change_label(self):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        try:
            self.currLabel = self.labelBox.get()
        except:
            pass

        strBox = self.objectListBox.get(0)
        strBox = strBox.split(':')[0] + ': ' + str(self.currLabel)
        self.objectListBox.delete(sel)
        self.objectListBox.insert(sel, strBox)
        self.objectLabelList.pop(sel[0])
        self.objectLabelList.insert(sel[0], self.currLabel)
        self.currLabel = None

    def del_bbox(self):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.canvas.delete(self.bboxIdList[idx])

        if self.bboxIdList[idx] in self.bboxOvalDict:
            oval = self.bboxOvalDict[self.bboxIdList[idx]]
            self.canvas.delete(oval[0])
            self.canvas.delete(oval[1])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)
        if self.objectListBox.size() > 0:
            self.objectListBox.select_set(0)
            self.label_select(None)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.ovalIdList)):
            self.canvas.delete(self.ovalIdList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.ovalIdList = []
        self.bboxOvalDict.clear()

    def add_label(self):
        if self.textBox.get() != '':
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if self.textBox.get() not in curr_label_list:
                self.labelListBox.insert(END, str(self.textBox.get()))
            self.textBox.delete(0, 'end')

    def add_labels_coco(self):
        for listidxcoco, list_label_coco in enumerate(self.cocoLabels):
            if self.cocoIntVars[listidxcoco].get():
                curr_label_list = self.labelListBox.get(0, END)
                curr_label_list = list(curr_label_list)
                if list_label_coco not in curr_label_list:
                    self.labelListBox.insert(END, str(list_label_coco))

    def add_all_classes(self):
        for label, name in self.cocoLabels.items():
            # if self.cocoIntVars[listidxcoco].get():
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            label_tip = label + ": " + name
            if label_tip not in curr_label_list:
                self.labelListBox.insert(END, str(label_tip))

        self.initHelpText.insert(END, 'a: x1 - 1')
        self.initHelpText.insert(END, 'd: x1 + 1')
        self.initHelpText.insert(END, 'w: y1 - 1')
        self.initHelpText.insert(END, 'z: y1 + 1')
        self.initHelpText.insert(END, 'j: x2 - 1')
        self.initHelpText.insert(END, 'l: x2 + 1')
        self.initHelpText.insert(END, 'i: y2 - 1')
        self.initHelpText.insert(END, 'm: y2 + 1')

    def automate(self):
        # self.processingLabel.config(text="Processing     ")
        # self.processingLabel.update_idletasks()

        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
        except:
            pass
        boxes = []
        label_list = []

        if len(self.bboxList) > 0 and len(self.templateList) > 0:
            for i in range(len(self.templateList)):
                match_box, ncc1 = template_match(self.img, self.templateList[i], self.bboxList[i])
                if len(match_box) > 0:
                    boxes.append(match_box)
                    label_list.append(self.objectLabelList[i])

        config_labels = config.labels_to_names
        self.add_boundingbox(boxes, label_list)

        self.processingLabel.config(text="Done              ")

    def add_boundingbox(self, boxes, label_list):
        self.clear_bbox()
        for k in range(len(boxes)):
            box = boxes[k]
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            b = np.asarray(box, np.int16)
            self.add_box(b, label_list[k])

        if self.objectListBox.size() > 0:
            self.objectListBox.select_clear(0, len(self.bboxIdList) - 1)
            self.objectListBox.select_set(len(self.bboxIdList) - 1)
            self.label_select(None)

    def add_box(self, box, label):
        self.currBboxColor = config.COLORS[len(self.bboxList) % len(config.COLORS)]
        box = self.limate_box(box)
        bboxId = self.canvas.create_rectangle(box[0], box[1],
                                              box[2], box[3],
                                              width=2,
                                              outline=self.currBboxColor)
        self.bboxList.append((box[0], box[1], box[2], box[3]))
        self.bboxIdList.append(bboxId)

        if str(label) == 'None':
            label_str = 'None'
        else:
            label_str = str(label).split(':')[0] + ':' + self.cocoLabels[str(label).split(':')[0]]
        self.objectLabelList.append(label_str)
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (box[0], box[1], box[2], box[3]) + ': ' + label_str)
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                      fg=self.currBboxColor)
        self.objectListBox.focus()
        return bboxId

    def limate_box(self, box):
        box[0] = min(max(0, box[0]), self.win_w)
        box[1] = min(max(0, box[1]), self.win_h)
        box[2] = min(max(0, box[2]), self.win_w)
        box[3] = min(max(0, box[3]), self.win_h)
        if box[0] > box[2]:
            x = box[0]
            box[0] = box[2]
            box[2] = x
        if box[1] > box[3]:
            y = box[1]
            box[1] = box[3]
            box[3] = y
        return box


if __name__ == '__main__':
    root = Tk()
    root.state("zoomed")
    root_w, root_h = root.maxsize()

    root.resizable(width=True, height=True)
    # imgicon = PhotoImage(file='icon.gif')
    # root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root, root_w, root_h)
    root.mainloop()
