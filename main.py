import cv2
import pygame

# تهيئة مكتبة الصوت
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")  # ضع مسار صوت الإنذار

# تحميل نموذج MobileNet-SSD
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")  # تأكد من توفر هذه الملفات
cap = cv2.VideoCapture(0)  # تشغيل الكاميرا

# فئات الكائنات التي سنتعرف عليها (أشخاص وسيارات)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # تمرير الصورة عبر الشبكة
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # تحليل النتائج
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # تجاهل الكائنات ذات الثقة الأقل من 20%
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["person", "car", "bus", "bicycle", "motorbike"]:
                # تحديد الكائن في الإطار
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # تشغيل صوت الإنذار
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()

    # عرض الفيديو
    cv2.imshow("Street Monitor", frame)

    # إنهاء التشغيل عند ضغط مفتاح 'q'
    if cv2.waitKey(10) == ord('q'):
        break

# إغلاق الكاميرا وإيقاف جميع العمليات
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
