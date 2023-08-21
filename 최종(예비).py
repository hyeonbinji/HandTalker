import cv2
import numpy as np
import json

file = np.genfromtxt('dataSet.txt', delimiter=',')
angleFile = file[:, :-1]
labelFile = file[:, -1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)


num_joints = 21
sample_input = np.random.rand(num_joints, 3).astype(np.float32)


v1 = sample_input[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
v2 = sample_input[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
v = v2 - v1
v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
angles = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
angles = np.degrees(angles)


data = np.array([angles], dtype=np.float32)
ret, results, neighbours, dist = knn.findNearest(data, 3)
predicted_idx = int(results[0][0])


gesture_mapping = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
    8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
    15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v',
    22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'spacing', 27: 'backspace', 28: '1',
    29: '2', 30: '3', 31: '4', 32: '5', 33: '6', 34: '7', 35: '8', 36: '9'
}

predicted_gesture = gesture_mapping.get(predicted_idx, "Unknown Gesture")
result_dict = {'result': predicted_gesture}

# Convert to JSON and print the result
result_json = json.dumps(result_dict)
print(result_json)