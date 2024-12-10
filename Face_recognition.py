import cv2
import face_recognition
import pyttsx3


# 登録する顔画像の読み込みとエンコーディング
known_image = face_recognition.load_image_file("C:/Users/4085667/Documents/Arduino/AMB82-MINI/person1.jpg")  # 登録する画像ファイルのパス
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_name = "Tadashi"  # 登録者の名前

#　音声読み上げ初期化
engine = pyttsx3.init()
# 利用可能な音声のリストを取得
voices = engine.getProperty('voices')
# 音声の種類を変更（例: 1番目の音声を使用）
engine.setProperty('voice', voices[1].id)

# Webカメラの映像を取得
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔の位置を検出
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (face_encoding, face_location) in zip(face_encodings, face_locations):
        # 登録した顔と比較
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding, 0.4)
        name = "Unknown"

        # 一致した場合の処理
        if matches[0]:
            name = known_face_name
            frase = "こんにちは、" + name + "さん。げんきにゃんか？"
            engine.say(frase)
            engine.runAndWait()

        else:
            frase = "誰かわかりません"
            engine.say(frase)
            engine.runAndWait()

        # 矩形を描画
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 結果を表示
    cv2.imshow('Face Recognition', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()