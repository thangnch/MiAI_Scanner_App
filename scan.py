# import các thư viện
import cv2

# Tên file ảnh đầu vào
input_image  = 'mydoc.jpg'

# Đọc ảnh
image = cv2.imread(input_image)

# Chuyển ảnh mày thành ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ ảnh xám để xóa noise
blur = cv2.blur(gray,(3,3))

# Tìm cạnh bằng Canny
edge = cv2.Canny(blur, 50, 300, 3)

# Hiển thị lên màn hình cho dễ xem
cv2.imshow("gray", gray)
cv2.imshow("blur", blur)
cv2.imshow("edge", edge)
cv2.waitKey()

# Tìm contours
cnts = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
import imutils
cnts = imutils.grab_contours(cnts)

# Sắp xếp theo diện tích giảm dần
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


# Lấy contour đầu tiên - to nhất
cnts = cnts[:1]

# Vẽ contour lên ảnh gốc cho trực quan
p = cv2.arcLength(cnts[0], True)
r = cv2.approxPolyDP(cnts[0], 0.02*p, True)
# cv2.drawContours(image, [r], -1, (0,0,255), 3)

# Show ảnh
cv2.imshow("Draw", image)
cv2.waitKey()

# Đầu tiên reshape cái ROI của chúng ta về (4,2) - 4 tọa độ, mỗi tọa độ gồm x,y
r = r.reshape(4,2)

# Tính toán 04 góc theo thứ tự trên trái, trên phải, dưới phải, dưới trái
import numpy as np
rect = np.zeros((4,2), dtype='float32')

# Ta tính tổng các tọa độ theo cột
# Điểm trên trái sẽ có tổng nhỏ nhất
# Điểm dưới phải sẽ có tổng lớn nhất
s = np.sum(r, axis=1)
rect[0] = r[np.argmin(s)] # Trên trái
rect[2] = r[np.argmax(s)] # Dưới phải

# Ta tính sự khác nhau giữa các tọa độ theo cột
# Trên phải sẽ ít khác biệt nhất
# dưới trái là khác biệt nhất
diff = np.diff(r, axis=1)
rect[1] = r[np.argmin(diff)]
rect[3] = r[np.argmax(diff)]

# Tính toán chiều rộng và chiều cao của văn bản
(tl, tr, br, bl) = rect

width1 = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
width2 = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
Width = max(int(width1), int(width2))

height1 = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
height2 = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
Height = max(int(height1), int(height2))


# Tọa độ mới của văn bản
new_rect = np.array([
    [0,0],
    [Width-1, 0],
    [Width-1, Height-1],
    [0, Height-1]], dtype="float32")

# Tinh toán ma trận transform
M = cv2.getPerspectiveTransform(rect, new_rect)

# Thực hiện xoay và crop
output = cv2.warpPerspective(image, M, (Width, Height))

# Show ảnh
cv2.imshow("Output",output)
cv2.waitKey()

# Chuyển thành xám
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

# Áp threshold

_, output_final = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

# Show hàng
cv2.imshow("Ouput", output_final)
cv2.waitKey()