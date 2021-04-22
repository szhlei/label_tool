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
from PIL import Image, ImageTk

# import miscellaneous modules
import os
import numpy as np
import zlib
import cv2

import multiprocessing
import pickle

# make sure the file is inside semi-auto-image-annotation-tool-master
import pathlib

import config

# cur_path = pathlib.Path(__file__).parent.absolute()
from extract import extract
from extract_label import extract_label
from templateMatch import template_match
from csv_edit import *
from label_file import label_names

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

cur_path = pathlib.Path(__file__).parent.absolute().as_posix()
sys.path.append(cur_path)
os.chdir(cur_path)

class MainGUI:
    def __init__(self, master):

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
        self.img_dep = None
        self.tkimg = None
        self.depth_tkimg = None
        self.imageDir = ''
        self.imageDirPathBuffer = ''
        self.imageList = []
        self.imageTotal = 0
        self.imageCur = 0
        self.cur = 0
        self.bboxIdList = []
        self.bboxIdList_depth = []
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
        self.thresh = 0.5
        self.org_h = 0
        self.org_w = 0
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
        self.ctrlPanel = Frame(self.frame, height=980)
        self.ctrlPanel.grid(row=0, column=0, sticky=W + N)
        self.openBtn = Button(self.ctrlPanel, text='Open', command=self.open_image)
        self.openBtn.grid(columnspan=2, sticky=W + E)
        self.openDirBtn = Button(self.ctrlPanel, text='Open Dir', command=self.open_image_dir)
        self.openDirBtn.grid(columnspan=2, sticky = W + E)

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
        self.disp = Label(self.ctrlPanel, text='Coordinates:')
        self.disp.grid(columnspan=2, sticky=W + E)

        self.initHelpText = Listbox(self.ctrlPanel, exportselection = False)
        self.initHelpText.grid(columnspan=2, sticky=W + E)

        self.zoomPanelLabel = Label(self.ctrlPanel, text="Precision View Panel")
        self.zoomPanelLabel.grid(columnspan=2, sticky=W + E)
        self.zoomcanvas = Canvas(self.ctrlPanel, width=400, height=400)
        self.zoomcanvas.grid(columnspan=2, sticky=W + E)

        # Image Editing Region
        self.canvas = Canvas(self.frame, width=800, height=980)
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
        self.objectListBox = Listbox(self.listPanel, width=40, exportselection = False)
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

        self.labelListBox = Listbox(self.listPanel, exportselection = False)
        self.labelListBox.pack(fill=X, side=TOP)

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
        bboxes, label_list = get_previous(self.anno_filename, self.imageList[self.cur])
        self.add_boundingbox(bboxes, label_list)

    def load_image(self, file):
        self.img = Image.open(file)

        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(text='  ||   Image Number: %d / %d' % (self.imageCur, self.imageTotal))
        # Resize to Pascal VOC format
        w, h = self.img.size
        self.org_w, self.org_h = self.img.size

        self.click = False

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)

        img_dep = np.zeros((480, 640))

        file_ext = '.dat'
        if os.path.exists(os.path.splitext(file)[0] + '.pkl'):
            file_ext = '.pkl'

        with open(os.path.splitext(file)[0] + file_ext, 'rb') as f:
            if file_ext == '.pkl':
                # data = np.frombuffer(zlib.decompress(f.read()), dtype=np.uint16)
                data = np.frombuffer(zlib.decompress(pickle.load(f)), dtype=np.uint16)
            else:
                data = np.frombuffer(zlib.decompress(f.read()), dtype=np.uint16)
            dep_size = (480, 640)
            if data.shape[0] == 256000:
                dep_size = (400, 640)
            image_dep = data.reshape(dep_size)
            image_dep = cv2.equalizeHist(cv2.convertScaleAbs(image_dep, alpha=255 / image_dep.max()))
            img_dep[0:dep_size[0], :dep_size[1]] = image_dep
            # depth = cv2.add(np.asarray(self.img.convert('L')), image_dep)
            self.img_dep = Image.fromarray(img_dep)
        # depth = self.img.convert('L') * 0.5 + self.img_dep * 0.5
        self.depth_tkimg = ImageTk.PhotoImage(self.img_dep) #Image.fromarray(depth)
        self.canvas.create_image(0, 500, image=self.depth_tkimg, anchor=NW)

    def step_to(self):
        frame = int(self.stepBox.get())
        if frame >= 0 and frame < self.imageTotal:
            self.cur = frame
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
            bboxes, label_list = get_previous(self.anno_filename, self.imageList[self.cur])
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
            bboxes, label_list = get_previous(self.anno_filename, self.imageList[self.cur])
            self.add_boundingbox(bboxes, label_list)

        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def open_next_not_automate(self, event=None):
        self.save()
        if self.cur < self.imageTotal - 1:
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
            bboxes, label_list = get_previous(self.anno_filename, self.imageList[self.cur])
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
        elif event.char in ['a','w','d','z', 'j','i','l','m']:
            sel = self.objectListBox.curselection()
            if len(sel) != 1:
                return
            idx = sel[0]
            bboxId = self.bboxIdList[idx]
            bboxId_dep = self.bboxIdList_depth[idx]

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
            self.canvas.coords(bboxId_dep, (x1, 500 + y1, x2, 500 + y2))
            self.bboxList.pop(idx)
            self.bboxList.insert(idx, (x1, y1, x2, y2))

            self.objectListBox.delete(idx)
            self.objectListBox.insert(idx, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + self.objectLabelList[idx])
            self.objectListBox.select_clear(0, len(self.bboxIdList) - 1)
            self.objectListBox.select_set(idx)

    def save(self):
        img = self.img.copy()
        depth = self.img_dep.copy()
        self.templateList = []
        self.depth_templateList = []
        if len(self.bboxList) > 0:
            for idx, item in enumerate(self.bboxList):
                x1, y1, x2, y2 = self.bboxList[idx]
                self.templateList.append(img.crop((x1, y1, x2, y2)))
                self.depth_templateList.append(depth.crop((x1, y1, x2, y2)))
            # _thread.start_new_thread(self.save_thread, (img,))
            # p = multiprocessing.Process(target=self.save_thread, args=(img,))  # 申请子进程
            # p.start()
            self.saveProcess(self.anno_filename, self.imageList[self.cur], self.imageDirPathBuffer, self.bboxList
                             , self.objectLabelList)
        else:
            anno_file = self.anno_filename + '/' + os.path.splitext(self.imageList[self.cur])[0] + '.csv'
            if os.path.exists(anno_file):
                os.remove(anno_file)

    def saveProcess(self, anno_filename, image, imageDirPathBuffer, bboxList, objectLabelList):
        try:
            annotation_file = open(anno_filename + '/' + os.path.splitext(image)[0] + '.csv', 'w', encoding='utf-8')

            for idx, item in enumerate(bboxList):
                x1, y1, x2, y2 = bboxList[idx]
                annotation_file.write(
                    imageDirPathBuffer + '/' + image + ',' + ','.join(
                        map(str, bboxList[idx])) + ','
                    + str(objectLabelList[idx].split(':')[0]) + '\n')

            annotation_file.close()
        except:
            print('error')
            traceback.print_exc(file=sys.stdout)

    def mouse_click(self, event):

        ox1, oy1 = (event.x - 3), (event.y - 3)
        ox2, oy2 = (event.x + 3), (event.y + 3)

        oval_id = self.canvas.create_oval(ox1, oy1, ox2, oy2, fill=config.COLORS[len(self.bboxList) % len(config.COLORS)])
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

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        self.zoom_view(event)

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            self.zoomImgCrop = self.zoomImg.crop(((event.x - 40), (event.y - 40), (event.x + 40), (event.y + 40)))
            self.zoomImgCrop = self.zoomImgCrop.resize((400, 400))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 200, 400, 200, width=2)
            vl = self.zoomcanvas.create_line(200, 0, 200, 400, width=2)
        except:
            pass

    # def update_bbox(self):
    #     idx = self.bboxIdList.index(self.editbboxId)
    #     self.bboxIdList.pop(idx)
    #     self.bboxList.pop(idx)
    #     self.objectListBox.delete(idx)
    #     self.currLabel = self.objectLabelList[idx]
    #     self.objectLabelList.pop(idx)
    #     idx = idx*4


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
        self.canvas.delete(self.bboxIdList_depth[idx])

        if self.bboxIdList[idx] in self.bboxOvalDict:
            oval = self.bboxOvalDict[self.bboxIdList[idx]]
            self.canvas.delete(oval[0])
            self.canvas.delete(oval[1])
        self.bboxIdList.pop(idx)
        self.bboxIdList_depth.pop(idx)
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)
        if self.objectListBox.size() > 0:
            self.objectListBox.select_set(0)
            self.label_select(None)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
            self.canvas.delete(self.bboxIdList_depth[idx])
        for idx in range(len(self.ovalIdList)):
            self.canvas.delete(self.ovalIdList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxIdList_depth = []
        self.bboxList = []
        self.objectLabelList = []
        self.ovalIdList = []
        self.bboxOvalDict.clear()

    def add_label(self):
        if self.textBox.get() is not '':
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
        open_cv_image = np.array(self.img)
        # Convert RGB to BGR
        # opencvImage = open_cv_image[:, :, ::-1].copy()
        # if tensorflow
        boxes = []
        label_list = []

        if len(self.bboxList) > 0 and len(self.templateList) > 0:
            for i in range(len(self.templateList)):
                box1, ncc1 = template_match(self.img, self.templateList[i], self.bboxList[i])
                box2, ncc2 = template_match(self.img_dep, self.depth_templateList[i], self.bboxList[i])
                match_box = []
                if len(box1) > 0 and len(box2) > 0:
                    if ncc1 < ncc2:
                        match_box = box2
                    else:
                        match_box = box1
                elif len(box1) > 0:
                    match_box = box1
                elif len(box2) > 0:
                    match_box = box2
                if len(match_box) > 0:
                    # same = False
                    # for j in range(len(boxes)):
                    #     box2 = boxes[j]
                    #     overlap = bbox_overlaps(box1, np.asarray(box2))
                    #     if overlap > 0.8:
                    #         same = True
                    #         crop = self.img.crop(box2)
                    #         ncc2 = calc_ncc(self.templateList[i], crop)
                    #         if ncc1 > ncc2:
                    #             boxes[j] = box1
                    #             label_list[j] = self.objectLabelList[i]
                    #             break
                    # if not same:
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
            '''self.bboxId = self.canvas.create_rectangle(b[0], b[1],
                                                       b[2], b[3],
                                                       width=2,
                                                       outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])
            self.bboxList.append((b[0], b[1], b[2], b[3]))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.objectLabelList.append(label_list[k])
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[0], b[1], b[2], b[3]) + ': ' + label_list[k])
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                          fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])'''
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
        bboxId_dep = self.canvas.create_rectangle(box[0], 500 + box[1],
                                              box[2], 500 + box[3],
                                              width=2,
                                              outline=self.currBboxColor)
        self.bboxIdList_depth.append(bboxId_dep)
        label_str = ''
        if str(label) == 'None':
            label_str = 'None'
        else:
            label_str = str(label).split(':')[0] + ':' +self.cocoLabels[str(label).split(':')[0]]
        self.objectLabelList.append(label_str)
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (box[0], box[1], box[2], box[3]) + ': ' + label_str)
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                      fg=self.currBboxColor)
        self.objectListBox.focus()
        return bboxId

    def limate_box(self, box):
        box[0] = min(max(0, box[0]), self.org_w)
        box[1] = min(max(0, box[1]), self.org_h)
        box[2] = min(max(0, box[2]), self.org_w)
        box[3] = min(max(0, box[3]), self.org_h)
        if box[0] > box[2]:
            x = box[0]
            box[0] = box[2]
            box[2] = x
        if box[1] > box[3]:
            y = box[1]
            box[1] = box[3]
            box[3] = y
        return box


def bbox_overlaps(box1, box2):
    '''
    ex_roi 用来回归的anchor
    gt_roi 每个anchor对应的ground truth
    在进行回归前，保证每个需要回归的anchor都有一个gtbox作为回归的目标
    计算dx，dy时，使用的是anchor和gtbox的中心点，比如中心点x方向距离/anchor的w
    计算dw，dh时，使用的是对数log形式 np.log(gt_widths / ex_widths)
    bbox_overlaps
        boxes_1: x1, y, x2, y2
        boxes_2: x1, y, x2, y2
    '''
    assert box1.shape[0] == 4 and box2.shape[0] == 4
    box = []
    box.append(box1)
    box.append(box2)
    box = np.asarray(box)
    bxmin = np.max(box[:, 0])
    bymin = np.max(box[:, 1])
    bxmax = np.min(box[:, 2])
    bymax = np.min(box[:, 3])
    bwidth = bxmax - bxmin
    bhight = bymax - bxmin
    inter = bwidth * bhight
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union

def calc_ncc(crop1, crop2):
    crop1 = crop1.resize((64, 64))
    crop1 = np.asarray(crop1, 'float')
    mean_crop1 = [np.mean(crop1[:, :, 0]), np.mean(crop1[:, :, 1]), np.mean(crop1[:, :, 2])]
    sub_crop1 = crop1 - mean_crop1
    square_crop1 = np.sum(np.multiply(sub_crop1, sub_crop1)) ** 0.5

    crop2 = crop2.resize((64, 64))
    crop2 = np.asarray(crop2, 'float')
    mean_crop2 = [np.mean(crop2[:, :, 0]), np.mean(crop2[:, :, 1]), np.mean(crop2[:, :, 2])]
    sub_crop2 = crop2 - mean_crop2
    square_crop2 = np.sum(np.multiply(sub_crop2, sub_crop2)) ** 0.5

    coor = np.sum(np.multiply(sub_crop1, sub_crop2))
    ncc = coor / (square_crop1 * square_crop2)

    return ncc

if __name__ == '__main__':
    root = Tk()
    # root.state("zoomed")
    root.resizable(width=True, height=True)
    # imgicon = PhotoImage(file='icon.gif')
    # root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root)
    root.mainloop()
