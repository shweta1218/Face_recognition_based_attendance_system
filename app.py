import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from datetime import date, datetime
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from flask import send_file
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import face_recognition
import dlib
from scipy.spatial import distance
import random



app = Flask(__name__)
app.secret_key = '181204'
client = MongoClient('mongodb://localhost:27017/')
db = client['attendance']
admin_collection = db['admin']
student_collection = db['students']
print("Connected to MongoDB")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        if admin_collection.find_one({"username": username}):
            flash('Admin already exists. Please login.')
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password)
        admin_collection.insert_one({"username": username, "password": hashed_password,"email": email})
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

def valid_admin_credentials(username, password):
    admin = admin_collection.find_one({"username": username})
    if admin:
        return check_password_hash(admin['password'], password)
    return False

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if valid_admin_credentials(username, password):
            session['logged_in'] = True
            session['admin_name'] = username
            return redirect(url_for('home'))
        else:
            flash("Incorrect credentials, please try again.")
    
    return render_template('login.html')

nimgs = 50
imgBackground = cv2.imread("background.png")
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/download_attendance')
def download_attendance():
    try:
        return send_file(f'Attendance/Attendance-{datetoday}.csv', as_attachment=True)
    except Exception as e:
        flash("Error downloading file: " + str(e))
        return redirect(url_for('home'))

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     prediction = model.predict(facearray)
#     return prediction[0]  # Return the predicted class

if not os.path.exists('static/face_recognition_model.pkl'):
    print("Model file not found.")

def load_test_data(test_userlist):
    test_faces = []
    test_labels = []

    for user in test_userlist:
        user_dir = os.path.join('static/faces', user)
        if os.path.isdir(user_dir):
            for imgname in os.listdir(user_dir):
                img_path = os.path.join(user_dir, imgname)
                try:
                    img = cv2.imread(img_path)
                    resized_face = cv2.resize(img, (50, 50))
                    test_faces.append(resized_face.ravel())
                    test_labels.append(user)  # Assuming user name is the label
                except Exception as e:
                    print(f"Error processing test image {img_path}: {e}")

    return np.array(test_faces), np.array(test_labels)

def train_model():
    faces = []
    labels = []
    userlist = [user for user in os.listdir('static/faces') if os.path.isdir(os.path.join('static/faces', user))]

    for user in userlist:
        user_dir = os.path.join('static/faces', user)
        if os.path.isdir(user_dir):
            for imgname in os.listdir(user_dir):
                img_path = os.path.join(user_dir, imgname)
                try:
                    img = face_recognition.load_image_file(img_path)
                    face_encoding = face_recognition.face_encodings(img)[0]
                    faces.append(face_encoding)
                    labels.append(user)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    if faces:
        np.save('static/face_encodings.npy', faces)
        np.save('static/face_labels.npy', labels)
        print("Face encodings saved successfully.")


def identify_face(face):
    known_faces = np.load('static/face_encodings.npy')
    known_labels = np.load('static/face_labels.npy')
    face = face.flatten()  # Ensure it is 1D
    
    # Comparing the face encoding to the known faces
    distances = face_recognition.face_distance(known_faces, face)

    best_match_index = np.argmin(distances)

    # Set a threshold for matching
    if distances[best_match_index] < 0.6:  # Adjust threshold as needed
        return known_labels[best_match_index]
    else:
        return "Unknown"


def add_attendance(name):
    if name == "Unknown":
        return
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, len(userlist)

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)


@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])  # Vertical
    B = distance.euclidean(eye[2], eye[4])  # Vertical
    # Calculate the Euclidean distance between the horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])  # Horizontal

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eye_aspect_ratio(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    if len(rects) == 0:  # No faces detected
        return 0.0  # Return a value that won't cause an error

    for rect in rects:
        shape = predictor(gray, rect)
        left_eye = [(shape.part(36).x, shape.part(36).y),
                    (shape.part(37).x, shape.part(37).y),
                    (shape.part(38).x, shape.part(38).y),
                    (shape.part(39).x, shape.part(39).y),
                    (shape.part(40).x, shape.part(40).y),
                    (shape.part(41).x, shape.part(41).y)]
        
        right_eye = [(shape.part(42).x, shape.part(42).y),
                     (shape.part(43).x, shape.part(43).y),
                     (shape.part(44).x, shape.part(44).y),
                     (shape.part(45).x, shape.part(45).y),
                     (shape.part(46).x, shape.part(46).y),
                     (shape.part(47).x, shape.part(47).y)]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Return average EAR
        return (left_ear + right_ear) / 2.0

    return 0.0  # In case no face is detected


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    # if 'face_recognition_model.pkl' not in os.listdir('static'):
    #     return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
    #                            totalreg=totalreg(), datetoday2=datetoday2, 
    #                            mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (86, 32, 251), 2)

            # Detecting EAR for anti-spoofing
            ear = detect_eye_aspect_ratio(frame)
            if ear < 0.25:  # Adjust threshold as needed
                cv2.putText(frame, "Eyes closed - Attendance not marked", (left, top - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            else:
                # Identify the face
                identified_person = identify_face(face_encoding)
                add_attendance(identified_person)  # Log attendance
                cv2.putText(frame, f'{identified_person}', (left, top - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        print(request.form)  # Add this line to inspect the form data
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        newuseremail = request.form['newuseremail']
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        if admin_collection.find_one({"newusername": newusername}):
            flash('student already exists.')
            return redirect(url_for('home'))


        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        i, j = 0, 0
        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        student_collection.insert_one({
            "username": newusername,
            "roll_number": newuserid,
            "email": newuseremail,
            "photo_path": os.path.join(userimagefolder, f'{newusername}_0.jpg')
        })

        print('Training Model')
        train_model()

        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

    return render_template('add.html')



def send_email(to_email, subject, body, attachment):
    msg = MIMEMultipart()
    msg['From'] = 'shwetasawant1812@gmail.com'  # Replace with your email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Open the file as binary and attach it properly
        with open(attachment, "rb") as attachment_file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment_file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={attachment}')
            msg.attach(part)

        # Use correct SMTP server and port for Gmail
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('shwetasawant1812@gmail.com','livb wihc faau fpni')  # Use your actual password or app password
            server.send_message(msg)
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/email_attendance')
def email_attendance():
    try:
        csv_file = f'Attendance/Attendance-{datetoday}.csv'
        
        # Get admin email from the database
        admin = admin_collection.find_one({"username": session['admin_name']})
        if admin:
            admin_email = admin['email']
            print(f"Admin email found: {admin_email}")  # Add this to verify the admin's email
        else:
            admin_email = 'default@example.com'
            print("Admin not found, using default email")

        # Send the email with the CSV file
        send_email(admin_email, "Daily Attendance Report", "Please find the attached attendance report.", csv_file)

        return send_file(csv_file, as_attachment=True)
    except Exception as e:
        flash("Error downloading file: " + str(e))
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
