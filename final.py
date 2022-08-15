import math
from vehicle import Driver
from controller import Camera,Lidar
from controller import Speaker
import numpy as np 
import cv2
import warnings
from scipy.integrate import quad
from scipy.misc import derivative
import time
import threading
import pickle
warnings.filterwarnings("ignore")

driver = Driver()
timestep = int(driver.getBasicTimeStep())

camera = driver.getDevice("camera")
lidar = driver.getDevice("lidar")

Camera.enable(camera,timestep)
Lidar.enable(lidar,timestep)

cam_width = 1080 
cam_height = 720

lane_width = 710

ym_per_pix = 30 / cam_height
xm_per_pix = 3.7 / cam_width


Kp = 0.28
Ki = 0.11
Kd = -0.003

global left_lfollow,right_lfollow,turn_right,turn_left
global frame,direction_output,dt_traffic,parking,park_dev
global timer_check,timer_flag
global change_right,change_left,stop,speed,green_light
result = np.zeros((13,1))

with open('dt_traffic_last.pkl', 'rb') as f:
    dt_traffic = pickle.load(f)

def function(x):
    return x

def pid_controller(error,fin_time,start_time):
    P = -Kp * error
    (integral_value,_) = quad(lambda x: function(error),start_time,fin_time)
    I = -Ki * integral_value
    derivative_value = derivative(function,error,dx = (fin_time-start_time))
    D = -Kd * derivative_value 
    a = P + I + D
    return a
    
def perspectiveWarp(image):
    global cam_width,cam_height
    
    lane_points = np.float32([[200,570],[880,570],[5,700],[1075,700]])
    transform_points = np.float32([[0,0],[cam_width,0],[0,cam_height],[cam_width,cam_height]])

    matrix = cv2.getPerspectiveTransform(lane_points,transform_points)
    inverse_matrix = cv2.getPerspectiveTransform(transform_points,lane_points)
    
    result = cv2.warpPerspective(image,matrix,(cam_width,cam_height))
    
    return result,inverse_matrix

def proccesImage(image,thresh_min=150,thresh_max=255,kernel=(7,7)):
    
    thresh_canny1 = 40
    thresh_canny2 = 60
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_white, upper_white)
    hls_result = cv2.bitwise_and(image, image, mask = mask)

    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, thresh_min, thresh_max, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh,kernel, 0)
    canny = cv2.Canny(blur, thresh_canny1, thresh_canny2)
    result = cv2.add(thresh,canny)
    
    return result

def plotHistogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis = 0)
    midpoint = np.int(histogram.shape[0] / 2)
    return histogram,midpoint

def right_lane_follow(img,midpoint,histogram):
    global left_lfollow,right_lfollow,turn_right
    global timer_check
    try:
        right_cam_img = img[:,midpoint:]
        
        out_ind = np.transpose(np.nonzero(right_cam_img))
        
        x_coordinate = out_ind[:,0]
        y_coordinate = out_ind[:,1]
        
        y_coordinate = y_coordinate + midpoint
        
        right_fit = np.polyfit(x_coordinate,y_coordinate,2)
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
             
        rgt_x = np.trunc(right_fitx)
        
        lft_x = rgt_x - lane_width
        
        pts_right = np.array([np.transpose(np.vstack([rgt_x, ploty]))])
        pts_left = np.array([np.flipud(np.transpose(np.vstack([lft_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        mean_x = np.mean((lft_x, rgt_x), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
        
        mpts = pts_mean[-1][-1][-2].astype(int)
        pixelDeviation = img.shape[1] / 2 - abs(mpts)
        deviation = pixelDeviation * xm_per_pix
        
        print("Right Lane Following : {}".format(deviation))
        return deviation
    except:
        ticket = lidar_data()
        print("Lidar Signage detected:",ticket)
        if ticket == -1:
            deviation = 0
            print("Right Lane Disappeared : {}".format(deviation))
            return deviation
        else:
            turn_right = True
            timer_check = time.time()
            return -0.85
        
def left_lane_follow(img,midpoint,histogram):
    global left_lfollow,right_lfollow,turn_left
    global timer_check
    try:
        left_cam_img = img[:,:midpoint]
        
        out_ind = np.transpose(np.nonzero(left_cam_img))
        x_coordinate = out_ind[:,0]
        y_coordinate = out_ind[:,1]

        left_fit = np.polyfit(x_coordinate,y_coordinate,2)
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        
        lft_x = np.trunc(left_fitx)
        rgt_x = lft_x + lane_width
    
        pts_left = np.array([np.transpose(np.vstack([lft_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rgt_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        mean_x = np.mean((lft_x, rgt_x), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
        
        mpts = pts_mean[-1][-1][-2].astype(int)
        pixelDeviation = img.shape[1] / 2 - abs(mpts)
        deviation = pixelDeviation * xm_per_pix
        
        print("Left Lane Following : {}".format(deviation))
        return deviation
    except:
        ticket = lidar_data()
        print("Lidar Signage detected:",ticket)
        if ticket == -1:
            deviation = 0
            print("Left Lane Disappeared : {}".format(deviation))
            return deviation
        else:
            turn_left = True
            timer_check = time.time()
            return 0.85

def turn_of_right_algorithm():
    global turn_right,timer_check,direction_output
    global left_lfollow,right_lfollow
    print("Right Turn Algorithm")
    deviation = -0.85
    current_time = time.time()
    if current_time - timer_check > 4 :
        left_lfollow = True
        right_lfollow = False
        turn_right = False
        direction_output = 0
        print("Time is up")
    return deviation       
        
def turn_of_left_algorithm():
    global turn_left,timer_check,direction_output
    global left_lfollow,right_lfollow
    print("Left Turn Algorithm")
    deviation = 0.85
    current_time = time.time()
    if current_time - timer_check > 4 :
        left_lfollow = False
        right_lfollow = True
        turn_left = False
        direction_output = 0
        print("Time is up")
    return deviation

def change_lane(way) :
    global timer_flag,timer_check,left_lfollow, right_lfollow
    global change_right,change_left,direction_output
    timer_flag = True
    if way == "right" :
        print("Switch to the Right Lane")
        change_right = True
        current_time = time.time()
        if current_time - timer_check > 3:
            print("Time is up")
            timer_flag = False
            right_lfollow = True
            left_lfollow = False
            change_right = False
            direction_output = 0
        return -1.1
    else :
        print("Switch to the Left Lane")
        change_left = True
        current_time = time.time()
        if current_time - timer_check > 3:
            print("Time is up")
            timer_flag = False
            right_lfollow = False
            left_lfollow = True
            change_left = False
            direction_output = 0
        return 1.1
def lidar_data():
    temp_data = Lidar.getRangeImage(lidar)
    fix_data = []
    for data in temp_data:
        if data != float("inf"):
            fix_data.append(data)
    if len(fix_data) == 0:
        return -1
    mean = np.mean(fix_data)
    return mean

def passenger_stop():
    global stop,speed,timer_flag
    distance = lidar_data()
    timer_flag = True
    if distance != -1 and distance < 5:
        if stop == "durak" or stop == "dur":
            speed = 0
            current_time = time.time()
            if stop == "durak":
                print("Taking Passengers")
                if current_time - timer_check > 15:
                    print("Passengers Were Taken")
                    speed = 4
                    stop = "disable"
                    timer_flag = False
            else :
                print("Stop Sign")
                if current_time - timer_check > 10:
                    print("Stop Sign : Time is up")
                    speed = 4
                    stop = "disable"
                    timer_flag = False
        elif stop =="kirmizi_isik":
            speed = 0
            print("Red Light : Waiting")
            
def lane_detection(frame):
    global left_lfollow, right_lfollow,timer_check,timer_flag,direction_output,speed,green_light
    img,interval_matrix = perspectiveWarp(frame)
    img = proccesImage(img)
    histogram,midpoint = plotHistogram(img)
    
    #print(direction_output)
    # 0 -> ileri | 1 -> sola | 2 -> saga
    if stop == "durak" or stop == "kirmizi_isik" or stop == "dur":
       distance = lidar_data()
       if not timer_flag :
           timer_check = time.time()
       passenger_stop()
    elif green_light:
        print("Green Light: Go")
        speed = 4
        green_light = False
    if turn_left:
        return turn_of_left_algorithm()
    elif turn_right:
        return turn_of_right_algorithm()
    elif change_left or (direction_output == 1 and right_lfollow):
        if not timer_flag :
            timer_check = time.time()
        return change_lane("left")
    elif change_right or (direction_output == 2 and left_lfollow):
        if not timer_flag :
            timer_check = time.time()
        return change_lane("right")
    elif left_lfollow:
        deviation = left_lane_follow(img,midpoint,histogram)
        return deviation
    elif right_lfollow:
        deviation = right_lane_follow(img,midpoint,histogram)
        return deviation
    else : 
        deviation = 0
        print("Unexpected Error")
        return deviation

def traffic_sign_detection() :
    global stop,green_light
    threading.Timer(6.5, traffic_sign_detection).start()
    if len(frame) != 0:
        frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB = True, crop = False  )
        
        labels=['dur','durak','gecit_yok','ileri_saga','ileri_sola','kirmizi_isik','park','park_yasak','saga_donus_yok','saga_yon','sola_donus_yok','sola_yon','yesil_isik' ]
        
        dt_labels = ['saga_yon','sola_yon','ileri_saga','ileri_sola','sola_donus_yok','saga_donus_yok','gecit_yok']
        
        stop_labels = ["durak","kirmizi_isik","yesil_isik","dur"]        
        
        model = cv2.dnn.readNetFromDarknet("./yolov4-obj.cfg", "./2mayis.weights")
        
        layers = model.getLayerNames()
        
        output_layer = [layers[layer-1]for layer in model.getUnconnectedOutLayers()]
        
        model.setInput(frame_blob)
        detection_layers = model.forward(output_layer)
       
        ids_list = []
        all_sign = []
        for detection_layer in detection_layers:
            for object_detection in detection_layer:
                
                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence = scores[predicted_id]               
                if confidence > 0.80:
                    label = labels[predicted_id]
                    if label in stop_labels:
                        stop = label
                        if label == "yesil_isik":
                           green_light = True   
                    all_sign.append(label)
                    if label in dt_labels:
                        index = dt_labels.index(label)
                        ids_list.append(index)
        
        ids_list = np.unique(ids_list)
        all_sign = np.unique(all_sign)
        output = np.zeros(7)
        for i in ids_list:
            print(dt_labels[i])
            output[i] = 1
        dt_direction(output)

def parking_algorithm():
    print("Looking for Park Spot : ",park_dev)
    return park_dev / (cam_width/2.7) * -1

def dt_direction(array):
    global direction_output
    direction_output = dt_traffic.predict([array])

def initialize():
    global left_lfollow,right_lfollow,timer_check,timer_flag
    global direction_output,frame,change_right,change_left
    global parking,park_dev,turn_right,turn_left,stop,speed,green_light
    left_lfollow = False
    right_lfollow = True
    parking = False
    park_dev = 0
    timer_flag = False
    timer_check = 0
    direction_output = 2
    turn_right = False
    turn_left = False
    change_right = False
    change_left = False
    stop = "disable"
    green_light = False
    speed = 4
    frame = []
    traffic_sign_detection()
initialize()

while driver.step() != -1:
    global frame,speed
    driver.setCruisingSpeed(speed)
    start_time = time.time()

    camera_array = camera.getImageArray()
    camera_array = np.array(camera_array,np.uint8)
    camera_array = camera_array.transpose(1,0,2)
    camera_array = cv2.cvtColor(camera_array, cv2.COLOR_BGR2RGB)
    frame = camera_array
        
    dev = lane_detection(camera_array)
    fin_time = time.time() 
    rotation_angle = pid_controller(dev,fin_time,start_time)
    if rotation_angle > 0.6 :
        rotation_angle = 0.6
    elif rotation_angle < -0.6 :
        rotation_angle = - 0.6
    driver.setSteeringAngle(rotation_angle)
