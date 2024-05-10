from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')

# Training.
if __name__ == '__main__':
	results = model.train(
	   data='csgo.yaml',
	   imgsz=640,
	   epochs=100,
	   batch=16,
	   name='yolov8s_csgoV1_640')
	results = model.val()