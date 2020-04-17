import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("../yolov3-tiny.weights", "../yolov3-tiny.cfg")
classes = []
with open("../coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
#print(layer_names)
output_layers = [layer_names[layer[0] - 1]
                 for layer in net.getUnconnectedOutLayers()]

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

frame_id = 0
start_time = time.time()

while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    #Decting object
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]   #Because detection[0:5] are the info of the boxes
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for index in range(len(boxes)):
        if index in indexes:
            x, y, width, height = boxes[index]
            confidence = confidences[index]
            label = str(classes[class_ids[index]]) + " : " + str(round(confidence, 4))
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + width, x + height), color, 1)
            cv2.putText(frame, label, (x, y + 10), font, 0.5, color, 1)

    #Counting FPS
    costed_time = time.time() - start_time
    fps = round(frame_id / costed_time, 2)
    cv2.putText(frame, "FPS : " + str(fps), (10, 40), font, 2, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyWindow()