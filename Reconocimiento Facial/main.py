import cv2
import face_recognition

# Carga la imagen a comparar
image = cv2.imread("Images/Foto.png")

# Obtiene las coordenadas de la cara en la imagen
face_loc = face_recognition.face_locations(image)[0]

# Codifica la imagen de la cara
face_image_encodings = face_recognition.face_encodings(image, known_face_locations = [face_loc])[0]
print('face_image_encodings: ', face_image_encodings)

("Prueba de dibujo de rectángulo alrededor de la cara\n"
 "cv2.rectangle(image, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0))\n"
 "cv2.imshow(\"IMAGE\", image)\n"
 "cv2.waitKey(0)\n"
 "cv2.destroyAllWindows()")

# Inicializa la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Bucle de captura de video
while True:
    # Lee un frame de video
    ret, frame = cap.read()

    # Si no se puede leer el frame, termina el bucle
    if ret == False: break

    # Invierte el frame para una mejor visualización
    frame = cv2.flip(frame, 1)

    # Detecta las caras en el frame
    face_locations = face_recognition.face_locations(frame, model = "cnn")

    # Si hay caras detectadas
    if face_locations != []:
        # Itera sobre las caras detectadas
        for face_location in face_locations:
            # Codifica la cara en el frame
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations = [face_location])[0]

            # Compara la cara en el frame con la imagen codificada previamente
            result = face_recognition.compare_faces([face_image_encodings], face_frame_encodings, tolerance = 0.5)
            print("Result:", result)

            # Si se encuentra una coincidencia, muestra el nombre "Sebastian"
            if result[0] == True:
                text = "Sebastian"
                color = (125, 220, 0)
            # Si no se encuentra una coincidencia, muestra "Desconocido"
            else:
                text = "Desconocido"
                color = (50, 50, 255)

            # Dibuja un rectángulo alrededor de la cara en el frame
            cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color,
                          -1)
            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)

            # Muestra el nombre de la persona en el frame
            cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)
            cv2.imshow("Frame", frame)

            # Espera a que se pulse una tecla
            k = cv2.waitKey(1)
            # Si se pulsa la tecla ESC, termina el bucle
            if k == 27 & 0xFF:
                break
            # Libera la captura de video
        cap.release()
        # Cierra todas las ventanas
        cv2.destroyAllWindows()
