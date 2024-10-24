import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image   

EMOTION_CLASSES = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(), 
    transforms.Normalize(0.5, 0.5) 
])

def preprocess_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = preprocess(face_image)
    face_image = face_image.unsqueeze(0)
    return face_image

def predict_expression(model):
    capture = cv2.VideoCapture(0)
    while True:
        success, frame = capture.read()
        if not success:
            break
    
        frameBW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(frameBW, scaleFactor = 1.1, minNeighbors= 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), 2)

            y1 = y
            y2 = y + h
            x1 = x
            x2 = x + w
            crop_face = frameBW[y1:y2, x1:x2]
            face_tensor = preprocess_face(crop_face)
            face_tensor = face_tensor.to(device)
            with torch.no_grad():
                output = model(face_tensor)
                prediction_index = torch.argmax(output, dim=1).item()
                predicted_emotion = EMOTION_CLASSES[prediction_index]
            
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("emotionCNN", frame)

        if cv2.waitKey(1) & 0xFF == ord("1"):
            break
    
    capture.release()
    cv2.destroyAllWindows()