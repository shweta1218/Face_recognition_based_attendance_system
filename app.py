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
import csv
import time

app = Flask(__name__)
app.secret_key = ''
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

# if not os.path.exists('static/face_recognition_model.pkl'):
#     print("Model file not found.")

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
                    if img is None:
                        print(f"Unsupported image type or corrupted image: {img_path}")
                        continue
                    resized_face = cv2.resize(img, (50, 50))
                    resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
                    test_faces.append(resized_face.ravel())
                    test_labels.append(user)  # user name is the label
                except Exception as e:
                    print(f"Error processing test image {img_path}: {e}")

    return np.array(test_faces), np.array(test_labels)

def train_model():
    faces = []
    labels = []
    userlist = [user for user in os.listdir('static/faces') if os.path.isdir(os.path.join('static/faces', user))]

    if not userlist:
        print("No users found in the faces directory")
        return

    for user in userlist:
        user_dir = os.path.join('static/faces', user)
        if os.path.isdir(user_dir):
            for imgname in os.listdir(user_dir):
                img_path = os.path.join(user_dir, imgname)
                try:
                    # Load image using OpenCV
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Use OpenCV's face detector
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_locations = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
                    
                    if len(face_locations) > 0:
                        # Get the first face
                        x, y, w, h = face_locations[0]
                        
                        # Extract face region
                        face_image = img_rgb[y:y+h, x:x+w]
                        
                        # Resize to a standard size
                        face_image = cv2.resize(face_image, (150, 150))
                        
                        # Convert to grayscale for simpler processing
                        face_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
                        
                        # Flatten the image for storage
                        face_flat = face_gray.flatten()
                        
                        faces.append(face_flat)
                        labels.append(user)
                        print(f"Successfully processed image: {img_path}")
                    else:
                        print(f"No face found in image: {img_path}")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    if faces:
        faces = np.array(faces)
        labels = np.array(labels)
        np.save('static/face_encodings.npy', faces)
        np.save('static/face_labels.npy', labels)
        print(f"Face encodings saved successfully. Total faces: {len(faces)}")
    else:
        print("No faces were successfully processed.")

def identify_face(face_image):
    try:
        # Check if face encodings file exists
        if not os.path.exists('static/face_encodings.npy') or not os.path.exists('static/face_labels.npy'):
            print("Face encodings or labels file not found")
            return "Unknown"
            
        known_faces = np.load('static/face_encodings.npy')
        known_labels = np.load('static/face_labels.npy')
        
        # Check if we have any known faces
        if len(known_faces) == 0:
            print("No known faces in the database")
            return "Unknown"
        
        # Flatten the input face image
        face_flat = face_image.flatten()
        
        # Calculate distances to all known faces
        distances = []
        for known_face in known_faces:
            # Use Euclidean distance
            distance = np.linalg.norm(face_flat - known_face)
            distances.append(distance)
        
        distances = np.array(distances)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]

        # Set a threshold for matching (much more lenient)
        if min_distance < 30000:  # Significantly increased threshold
            print(f"Face recognized with distance: {min_distance}")
            return known_labels[best_match_index]
        else:
            print(f"Best match distance: {min_distance} (threshold: 30000)")
            return "Unknown"
            
    except Exception as e:
        print(f"Error in identify_face: {str(e)}")
        return "Unknown"

def add_attendance(name):
    try:
        if name == "Unknown":
            print("Skipping attendance for Unknown face")
            return
            
        username, userid = name.split('_')
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Ensure the Attendance directory exists
        if not os.path.exists('Attendance'):
            os.makedirs('Attendance')
            print("Created Attendance directory")
            
        # Ensure the CSV file exists
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write('Name,Roll,Time')
            print(f"Created new attendance file: {csv_path}")
            
        # Read the current attendance data
        try:
            df = pd.read_csv(csv_path)
            print(f"Current attendance data: {len(df)} records")
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            # Create a new DataFrame if there's an error
            df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
            
        # Check if the user is already marked present
        if int(userid) not in list(df['Roll']):
            # Add the new attendance record
            with open(csv_path, 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
            print(f"Added attendance for {username} (ID: {userid}) at {current_time}")
            return True
        else:
            print(f"User {username} (ID: {userid}) already marked present today")
            return False
    except Exception as e:
        print(f"Error in add_attendance: {str(e)}")
        return False

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
    try:
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        
        # Check if the CSV file exists
        if not os.path.exists(csv_path):
            print(f"Attendance file not found: {csv_path}")
            return [], [], [], 0
            
        # Read the CSV file
        try:
            df = pd.read_csv(csv_path)
            print(f"Successfully read attendance file with {len(df)} records")
            return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
        except Exception as e:
            print(f"Error reading attendance file: {str(e)}")
            return [], [], [], 0
    except Exception as e:
        print(f"Error in extract_attendance: {str(e)}")
        return [], [], [], 0

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


@app.route('/start')
def start():
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Failed to open webcam")
            return redirect(url_for('home'))
            
        # Create a window for live preview
        cv2.namedWindow('Attendance System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance System', 800, 600)
        
        # Capture frames for a few seconds to allow user to position face
        start_time = time.time()
        faces_found = False
        attendance_marked = False
        
        while time.time() - start_time < 10:  # Run for 10 seconds
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use more lenient parameters for face detection
            face_locations = face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # More lenient scale factor
                minNeighbors=3,   # Fewer neighbors required
                minSize=(30, 30)  # Smaller minimum face size
            )
            
            # Add a message to help the user
            cv2.putText(display_frame, "Press 'q' to quit", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add a countdown timer
            remaining_time = int(10 - (time.time() - start_time))
            cv2.putText(display_frame, f"Time remaining: {remaining_time}s", (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if len(face_locations) > 0:
                faces_found = True
                # Process each detected face
                for (x, y, w, h) in face_locations:
                    # Extract face region
                    face_image = frame[y:y+h, x:x+w]
                    
                    # Resize to match training size (150x150)
                    face_resized = cv2.resize(face_image, (150, 150))
                    
                    # Convert to grayscale for recognition
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Identify the face using our simplified approach
                    name = identify_face(face_gray)
                    
                    # Draw rectangle and name
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display the name with a background for better visibility
                    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x, y - 30), (x + text_size[0], y), (0, 255, 0), -1)
                    cv2.putText(display_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    if name != "Unknown":
                        # Try to mark attendance
                        if add_attendance(name):
                            attendance_marked = True
                            # Show success message on the frame
                            cv2.putText(display_frame, "Attendance Marked!", (30, 110), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If no faces detected, show a message
                cv2.putText(display_frame, "No face detected", (30, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Attendance System', display_frame)
            
            # Save the current frame
            cv2.imwrite('static/captured_frame.jpg', display_frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if not faces_found:
            flash("No faces detected. Please try again.")
        elif not attendance_marked:
            flash("No recognized faces found. Please try again.")
        else:
            flash("Attendance marked successfully!")
            
        # Print the path to the CSV file for debugging
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        print(f"Attendance CSV file: {os.path.abspath(csv_path)}")
        if os.path.exists(csv_path):
            print(f"CSV file exists with size: {os.path.getsize(csv_path)} bytes")
            try:
                df = pd.read_csv(csv_path)
                print(f"CSV contents: {len(df)} records")
                print(df)
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
        else:
            print("CSV file does not exist!")
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print("Error in face recognition:", str(e))
        if 'cap' in locals():
            cap.release()
        if 'cv2' in locals():
            cv2.destroyAllWindows()
        flash(f"Error in face recognition: {str(e)}")
        return redirect(url_for('home'))


@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        print(request.form)  # Debugging: Check form data
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        newuseremail = request.form['newuseremail']
        userimagefolder = f'static/faces/{newusername}_{newuserid}'

        # Check if student already exists
        if os.path.exists(userimagefolder):
            flash('Student already exists.')
            return redirect(url_for('home'))

        if not os.path.exists(userimagefolder):
            os.makedirs(userimagefolder)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Error: Cannot access the webcam.")
            return redirect(url_for('home'))

        i, j = 0, 0
        nimgs = 10  # Number of images to capture
        captured_images = 0

        while captured_images < nimgs:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                flash("Error: Failed to capture a valid image from the camera.")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {captured_images}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

                if j % 5 == 0:
                    filename = f'{newusername}_{captured_images}.jpg'
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Convert to RGB format
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    # Save as PNG instead of JPG to avoid compression issues
                    cv2.imwrite(os.path.join(userimagefolder, filename.replace('.jpg', '.png')), face_img_rgb)
                    captured_images += 1
                    print(f"Captured image {captured_images}/{nimgs}")

                j += 1

            cv2.imshow('Adding New User', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save user info in database
        student_collection.insert_one({
            "username": newusername,
            "roll_number": newuserid,
            "email": newuseremail,
            "photo_path": os.path.join(userimagefolder, f'{newusername}_0.png')
        })

        print('Training Model...')
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
            server.login('shwetasawant1812@gmail.com','')  # Use your actual password or app password
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

CSV_DIRECTORY = "D:/TYDS/face_recognition_flask/Attendance"

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        date_str = request.args.get('date')
        names, rolls, times, l = [], [], [], 0

        if date_str:
            # Convert the date input (YYYY-MM-DD) to the desired format (MM_DD_YY)
            date_parts = date_str.split('-')
            formatted_date = f"Attendance-{date_parts[1]}{date_parts[2]}{date_parts[0][2:]}.csv"
            file_path = os.path.join(CSV_DIRECTORY, formatted_date)

            # Debugging print to check the file path
            print("Looking for file at:", file_path)

            # Check if the CSV file for the entered date exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip the header row
                    for row in reader:
                        names.append(row[0])
                        rolls.append(row[1])
                        times.append(row[2])
                l = len(names)
                return render_template('home.html', names=names, rolls=rolls, times=times, l=l, datetoday2=date_str)
            else:
                flash(f"Attendance record for {date_str} not found.")

        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, datetoday2="")
if __name__ == '__main__':
    app.run(debug=True)
