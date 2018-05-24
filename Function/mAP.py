
CLASSES = ('__background__','jyz', 'xjyz', 'dwq', 'xj',
          'cls', 'tgsr', 'xzsr', 'xlxj')


def IOU(vertice1, vertice2):  # ver1:[xmin, ymin, w,h], verticel2:[xmin, ymin, xmax, ymax]
    lu = np.maximum(vertice1[0:2], vertice2[0:2])
    rd = np.minimum(vertice1[2:4], vertice2[2:4])
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[0] * intersection[1]
    square1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    square2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    return np.clip(inter_square / union_square, 0.0, 1.0)


def load_annotation(image_name):
    image_name = image_name.split('.')[0]
    xml_file = os.path.join(cfg.DATA_DIR,'VOCdevkit2007','VOC2007', 'Annotations',image_name+'.xml')
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    labels = {}
    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        class_name = obj.find('name').text.lower().strip()
        if class_name != 'left':
            cls = obj.find('name').text.lower().strip()
            boxes = [x1, y1, x2, y2]
            labels[tuple(boxes)] = cls
    return labels

def nms():
	pass
def im_detect():
	pass
	
def test_tool(net, image_set):
    cls_pre_AP = {}
    cls_gt_num={}
    cls_pre_num={}
    APs = {}
    for class_name in CLASSES:
        cls_pre_AP[class_name] =[[],[]]
        cls_gt_num[class_name] = 0
        cls_pre_num[class_name] =0
        APs[class_name] = []
    for image_name in image_set:
        im_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007', 'VOC2007', 'ImageSets', image_name+'.jpg')
        xml_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007', 'VOC2007', 'Annotations', image_name + '.xml')
        im = cv2.imread(im_file)

        #score:[pre_num, cls_num]
        #boxes:[pre_num, cls_num*4]
        scores, boxes = im_detect(net, im)

        #gt_label:{tuple(box):cls}
        gt_label = load_annotation(xml_file)
		
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
			#dets:[pre_num, 5]
            keep = nms(dets, 0.3)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= 0.8)[0]

            final_predict = dets[inds, :]
            final_score = dets[inds, -1]
            final_ture = []
            cls_pre_num[cls] += len(final_score)
            for i in range(len(final_predict)) :
                box = final_predict[i, 0:4]
                for gt_box, gt_cls in gt_label.items():
                    if gt_cls == cls:
                        cls_gt_num[cls] += 1
                        gt_box = list(gt_box)
                        if IOU(gt_box, box) > 0.5:
                            final_ture.append(1)
                        else:
                            final_ture.append(0)

            cls_pre_AP[cls][0].extend(final_score)
            cls_pre_AP[cls][1].extend(final_ture)
			#cls_pre_AP-- {'jyz':[[0.991,0.98,0.99,0.96....],[1, 1, 0....]], 'xj':[[],[]]...}
    #print(cls_pre_AP)
    cls_APs = {}
    for key, value  in cls_pre_AP.items():
        cls_AP = {}
        cls_score = value[0]
        cls_true = value[1]
        for index, score in enumerate(cls_score):
            cls_AP[score] = cls_true[index]
        cls_APs[key] = sorted(cls_AP.items(),key = lambda x:x[0],reverse = True)
    #cls_APs--{'jyz':[(0.991,1),(0.99,0)...],'xj':[()]...}
    for key, value in cls_APs.items():
        positive = 0
        recall_target = 0.1
        for index,i in enumerate(value):
            true_or_fasle = i[1]
            if true_or_fasle == 1 :
                positive += 1
                recall = float(positive) / cls_gt_num[key]
                precision = float(positive) / (index +1)
                if recall > recall_target and recall < 1:
                    APs[key].append(precision)
                    recall_target += 0.1
        print(key,'r-ecall:',recall)
	#APs--{'jyz':[0.45,0.55,0.75..]，’xj‘:[...]...}
    print(APs)







