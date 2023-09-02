import os.path
import datetime
import xlsxwriter
from ultralytics import YOLO
import cv2
import helper
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.65
GREEN = (0, 255, 0)
LINE_COORD = 320

# open the webcam and start the live stream
video_cap = cv2.VideoCapture(0)
writer = helper.create_video_writer(video_cap, "output.mp4v")

# load the pre-trained model tracker and initialize ...
model = YOLO("yolov8x.pt")
model.to('cuda')
tracker = DeepSort(max_age=5)
track_list = []
center_dict = {}
ingoers = 0
outgoers = 0
start_date = datetime.date.today()
current_time = datetime.datetime.now().time()
temp_hour = current_time.hour
hourly_in = 0
hourly_out = 0

while True:

    # creating an excel file with todays date
    current_date = datetime.date.today()

    # check if a day is passed
    if current_date != start_date:
        workbook.close()
        start_date = current_date

    # if todays file dont exist, create it
    if not os.path.isfile(f"Data/PeopleCount_{current_date}.xlsx"):
        workbook = xlsxwriter.Workbook(f'Data/PeopleCount_{current_date}.xlsx')
        worksheet = workbook.add_worksheet("My sheet")
        workbook.close()
        workbook = xlsxwriter.Workbook(f'Data/PeopleCount_{current_date}.xlsx')
        worksheet = workbook.add_worksheet("My sheet")


        # write the file contents
        worksheet.write('A1', "In")
        worksheet.write('B1', "Out")
        worksheet.write('D1', "Change")
        worksheet.write('G1', "Hour")
        worksheet.write('G2', "7.00-8.00")
        worksheet.write('G3', "8.00-9.00")
        worksheet.write('G4', "9.00-10.00")
        worksheet.write('G5', "10.00-11.00")
        worksheet.write('G6', "11.00-12.00")
        worksheet.write('G7', "12.00-13.00")
        worksheet.write('G8', "13.00-14.00")
        worksheet.write('G9', "14.00-15.00")
        worksheet.write('G10', "15.00-16.00")
        worksheet.write('G11', "16.00-17.00")
        worksheet.write('G12', "17.00-18.00")
        worksheet.write('G13', "18.00-19.00")
        worksheet.write('G14', "19.00-20.00")
        worksheet.write('H1', "In")
        worksheet.write('I1', "Out")
        worksheet.write('K1', "Change")


    ret, frame = video_cap.read()

    # run the yolo model on the frame
    detections = model(frame, classes=0)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    # DETECTION

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence

        # check if detected obj. is person
        if int(data[5]) == 0:
            pass
        else:
            continue
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence])

    # TRACKING

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

        # check the tracks and add to the dictionary if not there
        if track not in track_list:
            track_list.append(track)
            center_dict[track] = []
        else:
            center_dict[track].append(center)

            # time at the moment
            current_time = datetime.datetime.now().time()
            current_time_str = f"{current_time.hour}:{current_time.minute}:{current_time.second}"

            # check current hour and control if we are still between same hours
            current_hour = current_time.hour
            if not current_hour == temp_hour:
                temp_hour = current_hour
                hourly_in = 0
                hourly_out = 0

            # check if center is passed the line
            if center[0] > LINE_COORD:
                if center_dict[track][len(center_dict[track]) - 2][0] < LINE_COORD:
                    worksheet.write(f'A{ingoers + 2}', current_time_str)
                    ingoers += 1
                    hourly_in += 1
                    worksheet.write(f'H{current_hour - 5}',hourly_in)
                    worksheet.write(f'K{current_hour - 5}', hourly_in - hourly_out)

            else:
                if center_dict[track][len(center_dict[track]) - 2][0] > LINE_COORD:
                    worksheet.write(f'B{outgoers + 2}', current_time_str)
                    outgoers += 1
                    hourly_out += 1
                    worksheet.write(f'I{current_hour - 5}',hourly_out)
                    worksheet.write(f'K{current_hour - 5}', hourly_in - hourly_out)

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.circle(frame, center, 2, (0, 0, 255), 3)

    # drawing the line
    cv2.line(frame, (LINE_COORD, 0), (LINE_COORD, 480), (255, 0, 0), 3)

    # writing the counts
    cv2.putText(frame, f"in: {ingoers}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 6)
    cv2.putText(frame, f"outs: {outgoers}", (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 6)
    cv2.putText(frame, f"Change: {ingoers - outgoers}", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 6)

    # write the total onto the file
    worksheet.write('D2', ingoers - outgoers)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        workbook.close()
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()

