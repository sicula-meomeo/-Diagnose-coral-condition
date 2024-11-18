import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.cm as cm

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('corals_classification_model.h5')

# Đọc và xử lý một ảnh duy nhất
img_path = 'datatest/test2.jpg'  # Đường dẫn tới ảnh của bạn
img_size = (224, 224)

# Hàm để đọc và tiền xử lý ảnh
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)  # Thêm một chiều cho batch
    return array

# Hàm tạo heatmap Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Hàm lưu và hiển thị Grad-CAM
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return cam_path

# Xác định lớp cuối cùng của mô hình (thay đổi theo mô hình của bạn)
last_conv_layer_name = "block5_conv4"

# Loại bỏ softmax khỏi lớp cuối cùng
model.layers[-1].activation = None

# Tiền xử lý ảnh
img_array = preprocess_input(get_img_array(img_path, size=img_size))

# Tạo heatmap Grad-CAM
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Lưu và hiển thị Grad-CAM
cam_path = save_and_display_gradcam(img_path, heatmap)

# Dự đoán nhãn cho ảnh
pred = model.predict(img_array)

# Nếu là phân loại nhị phân (2 lớp), dùng pred[0][0] để lấy xác suất cho lớp "healthy"
# Nếu là phân loại nhiều lớp, dùng tf.argmax để lấy chỉ số lớp có xác suất cao nhất
predicted_label = 'healthy' if pred[0][0] > 0.5 else 'bleached'  # Điều kiện cho phân loại nhị phân

# Nếu là phân loại đa lớp, bạn có thể sử dụng
# predicted_label = model.classes[tf.argmax(pred[0])]

# Hiển thị kết quả
plt.imshow(plt.imread(cam_path))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
