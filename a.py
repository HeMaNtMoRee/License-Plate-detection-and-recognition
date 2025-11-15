# from paddleocr import PaddleOCR
import time

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Run OCR (use .predict() to avoid deprecation warning)
# st=time.time()
# result = ocr.predict(r"C:\Users\BAPS\Downloads\number plate new en\number plate detection\download (1).jpg")
# et=time.time()

# # Iterate results (new dict-based format)
# st1=time.time()
# from paddleocr import PaddleOCR

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# results = ocr.predict(r"C:\Users\BAPS\Downloads\number plate new en\number plate detection\download (1).jpg")

# for res in results:   # each image
#     texts = res['rec_texts']
#     scores = res['rec_scores']
# results1 = []
# for text, score in zip(texts, scores):
#     results1.append(f"Detected: {text}, Confidence: {float(score):.2f}")
# print("\n".join(results1))


# et1=time.time()

# print("ocr time",et-st)
# print("print time",et1-st1)

# Initialize PaddleOCR instance
# from paddleocr import PaddleOCR
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_doc_unwarping=False,
    use_doc_orientation_classify=False
)


# # Run OCR inference on a sample image 
# st=time.time()
result = ocr.predict(
    input=r"D:\projects\number plate detection\plate\AJ3.png")
# et=time.time()

# # Visualize the results and save the JSON results
# st1=time.time()
for res in result:
    print(res['rec_texts'])
    print(res['rec_scores'])
# et1=time.time()
# print("ocr time",et-st)
# print("print time",et1-st1)    




