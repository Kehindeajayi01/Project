import argparse
import xml.etree.ElementTree as ET
import re
import os

#import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predDirectory", help = "path to the predicted file")
    parser.add_argument("--gtDirectory", help = "path to the groundtruth path")

    return parser

"""This function gets the root of the XML file"""
def getRoots(index):
    parser = get_args()
    args = parser.parse_args()
    gtDirectory = args.gtDirectory
    predDirectory = args.predDirectory

    gtFiles = os.listdir(gtDirectory)
    predFiles = os.listdir(predDirectory)

    # get an xml file from the ground truth file
    predFilePath = predFiles[index]
    #print(predFilePath)
    #check if ground truth file is found in prediction files
    if predFilePath in gtFiles:
        gtTree = ET.parse(os.path.join(gtDirectory, predFilePath))
        predTree = ET.parse(os.path.join(predDirectory, predFilePath))

        # get the roots of the trees
        gtRoot = gtTree.getroot()
        predRoot = predTree.getroot()

        return gtRoot, predRoot
    
"""This function extract the bounding boxes of the ground truth and prediction files"""
def getBoundingBox(index):
    gtRoot, predRoot = getRoots(index)

    gtBox, prBox = [], []  # TODO: remove
    # getting coordinates of bounding boxes
    for gt in gtRoot[0]:

        # This loop through the root children and extract the contents of the cell tag
        if gt.tag == 'cell':
            
            for pr in predRoot[0]:
                if pr.tag == 'cell':
                    cellAttrB = pr.attrib # a dictionary with keys like start_row, end_row, etc
                    start_rowB = cellAttrB['start-row']
                    end_rowB = cellAttrB['end-row']
                    start_colB = cellAttrB['start-col']
                    end_colB = cellAttrB['end-col']

                    cellAttrA = gt.attrib # a dictionary with keys like start_row, end_row, etc
                    start_rowA = cellAttrA['start-row']
                    end_rowA = cellAttrA['end-row']
                    start_colA = cellAttrA['start-col']
                    end_colA = cellAttrA['end-col']

                    # This check if the start_row, start_col, end_row, and end_col of both ground truth and predicted are aligned
                    if (start_rowB == start_rowA) and (end_rowB == end_rowA) and (start_colB == start_colA) and (end_colB == end_colA):
                        cellValuesA, cellValuesB = gt[0].attrib['points'], pr[0].attrib['points']
                        valA, valB = re.findall('\d*\d+', cellValuesA), re.findall('\d*\d+', cellValuesB)

                        xminA, xminB = int(valA[0]), int(valB[0])
                        yminA, yminB = int(valA[1]), int(valB[1])
                        xmaxA, xmaxB = int(valA[0]) + int(valA[4]), int(valB[0]) + int(valB[4])
                        ymaxA, ymaxB = int(valA[1]) + int(valA[3]), int(valB[1]) + int(valB[3])
                        bbox_gt = [xminA, yminA, xmaxA, ymaxA]
                        bbox_pr = [xminB, yminB, xmaxB, ymaxB]
                        gtBox.append(bbox_gt)
                        prBox.append(bbox_pr)
                       
                        

    return gtBox, prBox


def check_bbox(index, threshold):
    fp = 0; fn = 0; tp = 0
   
    gtBox, prBox = getBoundingBox(index)
    for gt, pr in zip(gtBox, prBox):
        iou = calc_iou(gt, pr)
        if iou > threshold:
            tp += 1
        # elif pr not in gtBox:
        #     fp += 1
        # elif gt not in prBox:
        #     fn += 1
        else:
            fp += 1
    return tp, fp


"""This function takes bounding boxes of ground truth and predictions and calculates the IOU"""
def calc_iou(boxA, boxB):
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    predDirectory = args.predDirectory

    predFiles = os.listdir(predDirectory)
    
    weighted_avg = 0
    for thresh in range(1, 10):
        threshold = thresh / 10
        tp = 0; fp = 0; 

        for index in range(len(predFiles)):
            tp_i, fp_i= check_bbox(index, threshold)
            tp += tp_i 
            fp += fp_i
            #fn += fn_i

        precision = tp / (tp + fp)
       # recall = tp / (tp + fn)

       # F1 = (2 * precision * recall) / (precision + recall)
        # print(f"F1: {round(F1, 3)}; precision: {round(precision, 3)}; recall: {recall}; threshold: {threshold} ")
        print(f"precision: {round(precision, 3)}; threshold: {threshold}")

