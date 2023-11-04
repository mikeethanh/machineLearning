# Thư viện caer và canaro dùng để xử lý và làm việc với hình ảnh trong ngữ cảnh của học máy và trí tuệ nhân tạo. Cụ thể:

# caer (Computer Vision in Python): Thư viện caer giúp trong việc xử lý và tiền xử lý dữ liệu hình ảnh, bao gồm việc chuyển đổi, 
# cắt, điều chỉnh kích thước, chuẩn hóa và làm sạch dữ liệu hình ảnh. Nó cung cấp nhiều công cụ để thực hiện các tác vụ này một cách dễ dàng và hiệu quả.

# canaro (Deep Learning Models): Thư viện canaro cung cấp một số mô hình học sâu đã được thiết kế sẵn, giúp bạn xây dựng mạng nơ-ron và mô hình 
# học máy dễ dàng. Nó bao gồm một loạt các mô hình và hàm tiện ích cho việc xây dựng, đào tạo và đánh giá mô hình học máy trong ngữ cảnh xử lý hình ảnh.

# Khi bạn làm việc với học máy và xử lý hình ảnh, việc sử dụng caer và canaro giúp bạn tiết kiệm thời gian và công sức trong việc tiền xử lý dữ liệu hình ảnh 
# và xây dựng các mô hình học máy cho các ứng dụng liên quan đến xử lý hình ảnh. Importing chúng cho phép bạn sử dụng các tính năng và chức năng mạnh mẽ mà các thư viện này cung cấp.
import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
# to_categorical: Đây là một hàm trong tensorflow.keras.utils dùng để chuyển đổi
#  nhãn số thành biểu diễn one-hot encoding. Khi đào tạo mô hình phân loại nhiều 
# lớp, bạn thường có nhãn dưới dạng số (ví dụ: 0, 1, 2, ...), nhưng mô hình yêu cầu 
# đầu ra dưới dạng biểu diễn one-hot encoding (ví dụ: [1, 0, 0], [0, 1, 0], [0, 0, 1], ...). 
# to_categorical được sử dụng để thực hiện chuyển đổi này.

# LearningRateScheduler: Đây là một callback trong tensorflow.keras.callbacks cho phép bạn động
#  thay đổi tốc độ học của mô hình trong quá trình đào tạo. Điều này có thể hữu ích khi bạn muốn 
# điều chỉnh tốc độ học dựa trên hiệu suất của mô hình. Chẳng hạn, bạn có thể giảm tốc độ học nếu 
# mô hình đạt đến một "điểm ngừng" hoặc tăng tốc độ học nếu mô hình cần thêm thời gian học.
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

IMG_SIZE = (80, 80)
channels = 1
char_path = r'E:\simpsons_dataset'

# Creating a character dictionary, sorting it in descending order
char_dict = {}
# Hàm os.listdir(char_path) trong Python được sử dụng để liệt kê tất cả các tệp và 
# thư mục trong một thư mục cụ thể được chỉ định bởi đường dẫn char_path. Hàm này 
# trả về một danh sách chứa tên của tất cả các tệp và thư mục trong thư mục được chỉ định.
for char in os.listdir(char_path):
    # os.path.join(char_path, char) được sử dụng để nối đường dẫn thư mục chứa tất cả
    #  các nhân vật char_path với tên của từng nhân vật char, tạo ra một đường dẫn hoàn 
    # chỉnh đến thư mục của nhân vật đó.

    # os.listdir(os.path.join(char_path, char)) sẽ trả về danh sách tất cả các tệp và thư mục
    #  có trong thư mục của nhân vật đó.

    # len(os.listdir(os.path.join(char_path, char))) sẽ trả về số lượng phần tử (tệp và thư mục) 
    # trong danh sách, đại diện cho số lượng tệp ảnh hoặc thư mục con của nhân vật đó.

    # Sau khi thực hiện dòng mã này cho tất cả các nhân vật trong thư mục char_path, bạn sẽ 
    # có một từ điển char_dict chứa thông tin về số lượng ảnh hoặc thư mục con cho từng nhân vật.
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)

# Getting the first 10 categories with the most number of images
characters = char_dict[:10]

# Create the training data
# Dòng mã train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True) đang thực hiện việc tiền xử 
# lý dữ liệu huấn luyện từ thư mục char_path để chuẩn bị dữ liệu huấn luyện cho mô hình máy học.

# char_path là đường dẫn tới thư mục chứa dữ liệu của các nhân vật (trong trường hợp này, các nhân vật từ bộ phim hoạt hình The Simpsons).

# characters là danh sách các nhân vật bạn đã trích xuất trước đó từ thư mục char_path. Đây là danh sách các nhân vật mà bạn quyết định sử dụng để huấn luyện mô hình.

# channels chỉ định số lượng kênh màu cho hình ảnh, trong trường hợp này, là 1 (ảnh xám).

# IMG_SIZE là kích thước ảnh mà bạn muốn sử dụng cho việc huấn luyện, thông qua biến đã được định nghĩa trước đó.

# isShuffle=True cho biết liệu dữ liệu huấn luyện có nên được xáo trộn không. Nếu isShuffle=True, thì dữ liệu huấn luyện sẽ được xáo trộn ngẫu nhiên để đảm 
# bảo tính ngẫu nhiên và đa dạng trong việc huấn luyện mô hình.

# Cuối cùng, caer.preprocess_from_dir là một hàm từ thư viện caer, được sử dụng để nạp dữ liệu từ thư mục, thực hiện tiền xử lý trên dữ liệu hình ảnh và 
# trả về dữ liệu đã được tiền xử lý dưới dạng biến train. Cụ thể, nó nạp ảnh từ thư mục char_path dựa trên danh sách các nhân vật trong characters và áp 
# dụng các bước tiền xử lý cần thiết, chẳng hạn như chuyển đổi hình ảnh sang định dạng đúng, thay đổi kích thước, và xáo trộn dữ liệu nếu được chỉ định. 
# Sau đó, dữ liệu đã được tiền xử lý sẽ sẵn sàng để huấn luyện mô hình máy học.

# char_path: Đây là đường dẫn tới thư mục chứa dữ liệu của các nhân vật, trong trường hợp này, là các hình ảnh của các nhân vật từ bộ phim hoạt hình The 
# Simpsons. Việc truyền char_path vào hàm giúp hàm biết nơi để tìm kiếm dữ liệu hình ảnh.

# characters: Đây là danh sách các nhân vật bạn muốn sử dụng để huấn luyện mô hình máy học. Bằng cách truyền danh sách này vào hàm, bạn đang chỉ định cho
#  hàm rằng bạn muốn nạp dữ liệu hình ảnh cho những nhân vật cụ thể này. Hàm sẽ chỉ nạp dữ liệu cho các nhân vật trong danh sách characters và bỏ qua những nhân vật khác.

# Ví dụ, nếu bạn có một thư mục chứa hình ảnh của 20 nhân vật khác nhau, nhưng bạn chỉ muốn huấn luyện mô hình cho 5 nhân vật cụ thể, bạn có thể tạo một
#  danh sách characters chứa 5 tên nhân vật đó và truyền danh sách này vào hàm caer.preprocess_from_dir. Hàm sẽ chỉ nạp và tiền xử lý dữ liệu hình ảnh
#  cho các nhân vật trong danh sách characters, làm giảm thiểu việc tiêu tốn thời gian và tài nguyên cho việc nạp và tiền xử lý dữ liệu không cần thiết.
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# Number of training samples

# Tập dữ liệu train là một danh sách (list) chứa các mảng numpy, trong đó mỗi mảng numpy tương ứng với một ảnh đã được tiền xử lý. Dựa vào cách mà 
# caer.preprocess_from_dir thường hoạt động, train có thể được coi là một danh sách các mảng numpy 2D (mảng 2 chiều) với các phần tử trong danh sách lần lượt là các hình ảnh.

# Mỗi mảng numpy 2D trong danh sách train biểu diễn một hình ảnh và có kích thước tương tự nhau, chính là kích thước ảnh đã được thiết lập bởi biến IMG_SIZE.

# Nói cách khác, train là một danh sách của các hình ảnh đã được chuyển thành các mảng numpy 2 chiều, giúp chuẩn bị dữ liệu để huấn luyện mô hình máy học.
# Visualizing the data (OpenCV doesn't display well in Jupyter notebooks)

# Hình ảnh thường được biểu diễn dưới dạng ma trận 2 chiều (hoặc tensor 2 chiều). Trong một ma trận 2 chiều, mỗi phần tử của ma trận thể hiện giá trị pixel tương ứng tại
#  vị trí đó trong hình ảnh. Điều này cho phép máy tính dễ dàng xử lý và thực hiện các phép toán trên hình ảnh.

# Khi bạn chuyển hình ảnh thành một mảng numpy 2 chiều, bạn có thể thực hiện nhiều phép toán xử lý hình ảnh, như chuyển đổi kích thước hình ảnh, chuẩn hóa giá trị pixel,
#  và thậm chí là áp dụng các mô hình học máy lên dữ liệu hình ảnh này.
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap='gray')
plt.show()


# Separating the array and corresponding labels
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
# Trong ví dụ của bạn, việc tách dữ liệu đào tạo thành hai mảng NumPy featureSet và labels được thực hiện dựa trên cấu trúc tập dữ liệu mà bạn đã chuẩn bị trước đó. Hãy cùng xem xét các bước chính:

# Bước 1: Dữ liệu đào tạo đã được tải và tiền xử lý trong biến train. Biến train chứa danh sách các hình ảnh cùng với các nhãn tương ứng của chúng (nhãn đề cập đến nhân vật trong hình ảnh). Dữ liệu đào tạo nằm trong train ở dạng list, trong đó mỗi phần tử của list là một cặp (hình ảnh, nhãn).

# Bước 2: Bạn sử dụng hàm caer.sep_train() để tách dữ liệu đào tạo thành hai mảng riêng biệt, đó là featureSet và labels.

# featureSet chứa dữ liệu hình ảnh. Mỗi hình ảnh đã được tiền xử lý, chuyển thành mảng NumPy 2 chiều và chuẩn hóa.

# labels chứa các nhãn tương ứng với từng hình ảnh trong featureSet.



# Pixel là viết tắt của "picture element" (một phần tử hình ảnh) và là đơn vị cơ bản để mô tả một điểm ảnh trên hình ảnh số hoặc hình ảnh kỹ thuật số. 
# Một hình ảnh số là một tập hợp các pixel, và mỗi pixel đại diện cho một điểm trên hình ảnh. Gia trị pixel là giá trị của pixel tại vị trí đó trên hình
#  ảnh và thường được sử dụng để biểu diễn màu sắc hoặc độ sáng của điểm ảnh đó.

# Gia trị pixel thường được biểu diễn bằng một số nguyên hoặc số thực, tùy thuộc vào độ phân giải của hình ảnh và kiểu dữ liệu sử dụng. Trong hình ảnh 
# màu, mỗi pixel thường có ba giá trị cho các thành phần màu cơ bản (đỏ, xanh lá cây và xanh lam), và mỗi giá trị này có thể nằm trong khoảng từ 0 đến 
# 255 trong hệ thống màu RGB thông thường.

# Ví dụ, nếu bạn có một hình ảnh màu và bạn lấy giá trị pixel tại một điểm cụ thể trên hình ảnh, bạn sẽ nhận được một bộ ba giá trị (R, G, B) hoặc 
# (đỏ, xanh lá cây, xanh lam) đại diện cho màu sắc của điểm đó. Gia trị của mỗi thành phần màu trong pixel có thể là một số từ 0 đến 255, chỉ định mức
#  độ của màu tương ứng.

# trong ngữ cảnh của xử lý ảnh và đồ họa máy tính, "điểm ảnh" (pixel) là đơn vị cơ bản nhất của một hình ảnh số. Một điểm ảnh là một "điểm" trên lưới, 
# thường là một hình vuông nhỏ, trong một bức tranh số. Nó là đơn vị tối thiểu chứa thông tin về màu sắc và độ sáng của một điểm cụ thể trên hình ảnh.

# Mỗi điểm ảnh có thể được biểu diễn bằng một số, thường là một số nguyên, đại diện cho giá trị màu hoặc độ sáng tại điểm đó. Số lượng điểm ảnh trong
#  một hình ảnh ảnh hưởng đến độ phân giải của hình ảnh. Hình ảnh với nhiều điểm ảnh hơn thường có độ phân giải cao hơn, mang lại chi tiết hình ảnh tốt
#  hơn, nhưng cũng yêu cầu nhiều dữ liệu hơn để lưu trữ và xử lý.

# Normalize the featureSet ==> (0,1)
# Bước "Normalize the featureSet" có nhiệm vụ chuyển đổi giá trị của các điểm ảnh trong featureSet (tập dữ liệu hình ảnh) để đảm bảo rằng chúng nằm 
# trong khoảng giá trị nhất định, thường là từ 0 đến 1. Việc này thường được thực hiện để chuẩn hóa dữ liệu hình ảnh, làm cho các giá trị của các 
# điểm ảnh nằm trong phạm vi dễ quản lý và thích hợp cho việc đào tạo mô hình máy học.
featureSet = caer.normalize(featureSet)

# Converting numerical labels to binary class vectors
#  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
# Dòng code labels = to_categorical(labels, len(characters) trong đoạn code bạn đưa là để chuyển đổi các nhãn (labels) từ dạng số nguyên (integer) sang dạng biểu diễn one-hot encoding.

# Biểu diễn one-hot encoding là một phương pháp biểu diễn các nhãn (labels) trong các bài toán phân loại. Thay vì sử dụng giá trị số nguyên để biểu diễn lớp (class) của 
# một mẫu, ta tạo một vector có độ dài bằng số lớp trong bài toán. Vector này có giá trị bằng 1 tại vị trí tương ứng với lớp của mẫu và 0 tại các vị trí còn lại.

# Trong trường hợp của bạn, to_categorical(labels, len(characters)) chuyển đổi labels (đã là các số nguyên biểu diễn các lớp) thành biểu diễn one-hot encoding để sử dụng 
# cho việc đào tạo mô hình. Điều này đảm bảo rằng mô hình có thể hiểu được các lớp dưới dạng các vector one-hot encoding thay vì số nguyên và dễ dàng thực hiện các phép 
# toán so sánh và tính toán
labels = to_categorical(labels, len(characters))

# Creating train and validation data
# Dòng code x_train, x_val, y_train, y_val = caer.train_test_split(featureSet, labels, val_ratio=.2) được sử dụng để chia dữ liệu thành hai tập dữ liệu con: 
# một tập dữ liệu huấn luyện (training) và một tập dữ liệu kiểm tra (validation). Cụ thể:

# x_train chứa dữ liệu hình ảnh được sử dụng để huấn luyện mô hình máy học.

# y_train chứa các nhãn tương ứng với x_train.

# x_val chứa dữ liệu hình ảnh được sử dụng để kiểm tra và đánh giá mô hình trong quá trình huấn luyện.

# y_val chứa các nhãn tương ứng với x_val.

# Việc này là để đảm bảo rằng mô hình được đánh giá trên dữ liệu độc lập, không được sử dụng trong quá trình huấn luyện. val_ratio=.2 chỉ định tỷ lệ mà dữ liệu kiểm tra 
# chiếm trong tập dữ liệu gốc. Trong trường hợp này, tỷ lệ kiểm tra chiếm 20% tổng số dữ liệu.

# Việc này giúp kiểm tra khả năng tổng quát hóa của mô hình, đánh giá hiệu suất của mô hình trên dữ liệu nó chưa từng thấy trước đó.
# caer.train_test_split chia dữ liệu theo một tỷ lệ cụ thể được chỉ định. Trong trường hợp này, tỷ lệ là val_ratio=.2, tức là dữ liệu kiểm tra chiếm 20% tổng số dữ 
# liệu ban đầu và dữ liệu huấn luyện chiếm 80%.

# Việc chia dữ liệu thành hai phần, một cho huấn luyện và một cho kiểm tra, là một phần quan trọng của việc xây dựng và đánh giá mô hình máy học. Việc này giúp đánh
#  giá hiệu suất của mô hình trên dữ liệu nó chưa từng thấy trước đó và kiểm tra khả năng tổng quát hóa của mô hình.

# Mô hình được huấn luyện trên dữ liệu huấn luyện, sau đó kiểm tra hiệu suất trên dữ liệu kiểm tra để đảm bảo rằng nó hoạt động tốt trên dữ liệu mới. Khi bạn chia dữ 
# liệu thành hai tập dữ liệu, bạn có thể đảm bảo rằng dữ liệu kiểm tra không được sử dụng trong quá trình huấn luyện, giúp đánh giá hiệu suất mô hình một cách công bằng.
x_train, x_val, y_train, y_val = caer.train_test_split(featureSet, labels, val_ratio=.2)

# Deleting variables to save memory
del train
del featureSet
del labels
gc.collect()
# Sau khi thực hiện các dòng này, bộ nhớ đã được giải phóng và được sử dụng lại cho các mục đích khác, giúp tránh lãng phí tài nguyên hệ thống. gc.collect() 
# là một cuộc gọi để gom rác (garbage collection) để xóa các đối tượng không sử dụng khỏi bộ nhớ một cách hiệu quả.


# Useful variables when training
# BATCH_SIZE = 32: Đây là kích thước của các batch dữ liệu mà bạn sử dụng để huấn luyện mô hình. Batch là một tập hợp con của dữ liệu đào tạo được sử dụng để cập nhật 
# trọng số của mô hình trong mỗi vòng lặp. Batch size 32 có nghĩa rằng mỗi lần mô hình sẽ được cập nhật dựa trên 32 ví dụ từ dữ liệu đào tạo.

# EPOCHS = 10: Đây là số lần bạn muốn lặp lại toàn bộ tập dữ liệu đào tạo trong quá trình huấn luyện. Mỗi epoch là một lần lặp lại một lượt đầy đủ các dữ liệu đào tạo. 
# Số lượng epochs quyết định bao nhiêu lần mô hình sẽ học từ toàn bộ tập dữ liệu.

# Cả hai tham số này quyết định quá trình huấn luyện, cụ thể:

# BATCH_SIZE quyết định kích thước của mỗi lần cập nhật trọng số trong quá trình huấn luyện.
# EPOCHS quyết định bao nhiêu lần toàn bộ dữ liệu đào tạo sẽ được sử dụng trong quá trình huấn luyện.
# Chúng đều là các siêu tham số quan trọng trong quá trình huấn luyện mô hình và phải được thiết lập sao cho phù hợp với bài toán và dữ liệu cụ thể. Thông qua việc 
# điều chỉnh BATCH_SIZE và EPOCHS, bạn có thể kiểm soát quá trình huấn luyện, thời gian và hiệu suất của mô hình.

BATCH_SIZE = 32
EPOCHS = 10

# Image data generator (introduces randomness in network ==> better accuracy)
# Hàm canaro.generators.imageDataGenerator() được sử dụng để tạo một đối tượng của lớp ImageDataGenerator từ thư viện Keras, nhưng được tùy chỉnh cho việc tiền xử lý hình 
# ảnh trong bài toán của bạn.

# ImageDataGenerator là một công cụ mạnh mẽ trong Keras cho việc tạo ra các biến thể dữ liệu từ tập dữ liệu ban đầu bằng cách thực hiện các biến đổi hình ảnh như xoay, 
# phóng to, thu nhỏ, lật, và nhiều biến đổi hình ảnh khác. Điều này giúp tăng cường tập dữ liệu đào tạo và giúp mô hình học được các đặc trưng tổng quát hơn.

# Trong ví dụ của bạn, bạn đã sử dụng canaro.generators.imageDataGenerator() để tạo một đối tượng ImageDataGenerator. Điều này cho phép bạn sử dụng các phương thức của
#  đối tượng này để thực hiện các biến đổi hình ảnh ngẫu nhiên trong quá trình huấn luyện mô hình. Các biến đổi hình ảnh này giúp giảm overfitting và tạo ra sự đa dạng 
# trong dữ liệu đào tạo.

# Cách hoạt động của ImageDataGenerator là sẽ áp dụng các biến đổi này ngẫu nhiên cho từng hình ảnh trong mỗi batch trong quá trình huấn luyện, tạo ra các phiên bản biến 
# thể của dữ liệu để huấn luyện mô hình.
datagen = canaro.generators.imageDataGenerator()
# Hàm datagen.flow(x_train, y_train, batch_size=BATCH_SIZE) dùng để tạo ra một generator cho việc tạo dữ liệu đào tạo (training data generator) từ dữ liệu đào tạo ban đầu x_train 
# và nhãn tương ứng y_train. Generator này sẽ tạo ra các batch dữ liệu và nhãn cho việc huấn luyện mô hình.

# Cách hoạt động của phương thức flow là:

# Chia dữ liệu đào tạo và nhãn thành các batch có kích thước là BATCH_SIZE (số lượng mẫu trong mỗi batch).

# Dữ liệu và nhãn trong mỗi batch được truyền qua hàm generator để thực hiện các biến đổi dữ liệu ngẫu nhiên (nếu bạn đã cấu hình các biến đổi bằng ImageDataGenerator).

# Các batch dữ liệu và nhãn đã được xử lý sẽ được trả về trong quá trình huấn luyện mô hình. Hàm generator sẽ tạo ra liên tục các batch cho việc huấn luyện, giúp tăng cường
#  tập dữ liệu đào tạo và ngăn chặn mô hình khỏi overfitting.

# Cách sử dụng generator này thường được áp dụng trong các bài toán có tập dữ liệu lớn, để giảm bộ nhớ cần thiết để lưu toàn bộ tập dữ liệu trong RAM, và để tăng cường tính 
# ngẫu nhiên và đa dạng của dữ liệu đào tạo.

# Trong ngữ cảnh của lập trình và máy học, một generator (còn được gọi là iterator) là một cách để duyệt qua các phần tử của một tập dữ liệu mà không cần lưu trữ tất cả các
# phần tử đó trong bộ nhớ một lúc. Thay vào đó, generator sẽ tạo và trả về các phần tử từng cái một khi cần thiết.

# Điều này rất hữu ích khi bạn làm việc với các tập dữ liệu lớn mà không muốn tải toàn bộ dữ liệu lên RAM hoặc khi bạn muốn thực hiện xử lý từng phần tử một và tạo ra 
# các phần tử mới từ dữ liệu gốc mà không cần tạo ra toàn bộ dữ liệu mới.
# au khi training xong, train_gen chứa một generator. Generator này được sử dụng để tạo ra các batch dữ liệu từ tập dữ liệu huấn luyện x_train và y_train trong quá trình huấn luyện mô hình.

# Mỗi lần bạn gọi train_gen (thông qua phương thức flow), nó sẽ trả về một batch dữ liệu chứa BATCH_SIZE ví dụ và nhãn từ x_train và y_train. Generator này giúp tiết kiệm bộ
#  nhớ bằng cách tạo và trả về từng batch dữ liệu khi cần thiết, không cần lưu trữ toàn bộ tập dữ liệu trong RAM.

# rain_gen là một generator (bộ sinh) dùng để cung cấp dữ liệu huấn luyện từ tập dữ liệu x_train và nhãn y_train cho mô hình trong quá trình huấn luyện. Cụ thể, nó có các vai trò sau:

# Chia dữ liệu thành các batch: train_gen chia tập dữ liệu huấn luyện thành các batch có kích thước là BATCH_SIZE. Điều này giúp tối ưu hóa quá trình huấn luyện và tận dụng hiệu quả 
# các tài nguyên tính toán.

# Quản lý và trình bày dữ liệu: Nó quản lý việc lặp qua tập dữ liệu và cung cấp từng batch cho mô hình khi cần. Điều này giúp bạn không cần phải tải toàn bộ tập dữ liệu vào bộ nhớ 
# một lúc, điều này quan trọng khi bạn có một tập dữ liệu lớn.

# Cung cấp dữ liệu động: Khi bạn gọi next(train_gen), nó sẽ trả về một batch mới từ tập dữ liệu huấn luyện. Bằng cách này, bạn có thể huấn luyện mô hình trên dữ liệu mà bạn chưa nhìn
#  thấy trong quá trình huấn luyện trước đó (data augmentation).

# Việc sử dụng generator giúp giảm tải bộ nhớ và tối ưu hóa quá trình huấn luyện. Cách này đặc biệt hữu ích khi bạn có tập dữ liệu lớn mà không thể nạp hoàn toàn vào bộ nhớ.
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Create our model (returns the compiled model)
# Câu lệnh model = canaro.models.createSimpsonsModel(...) được sử dụng để tạo một mô hình mạng nơ-ron sử dụng cho bài toán phân loại các nhân vật trong loạt phim hoạt hình Simpsons. Hàm này trả về một đối tượng mô hình mạng nơ-ron được cấu hình theo các thông số bạn đã cung cấp. Hãy cùng tìm hiểu về từng tham số:

# IMG_SIZE: Kích thước ảnh đầu vào. Mô hình sẽ được cấu hình để xử lý ảnh có kích thước này.

# channels: Số lượng kênh (channels) của ảnh đầu vào. Trong trường hợp này, bạn đã đặt channels ban đầu là 1, cho biết rằng ảnh đầu vào là ảnh màu xám (grayscale).

# output_dim: Số lượng lớp đầu ra của mô hình, tức là số lượng nhãn (characters) mà bạn đang cố gắng phân loại.

# loss: Hàm mất mát mà mô hình sẽ sử dụng trong quá trình huấn luyện. Trong trường hợp này, bạn đặt loss là 'binary_crossentropy', cho biết rằng đây là một bài toán phân loại nhị phân.

# learning_rate: Tốc độ học (learning rate) của thuật toán tối ưu hóa. Đây là một siêu tham số quan trọng quyết định tốc độ học của mô hình trong quá trình huấn luyện.

# momentum: Tham số momentum trong thuật toán tối ưu hóa. Nó điều chỉnh việc cập nhật trọng số của mô hình trong mỗi bước huấn luyện.

# nesterov: Tham số nesterov trong thuật toán tối ưu hóa. Khi nesterov là True, nó sử dụng phương pháp tối ưu hóa Nesterov Accelerated Gradient (NAG).

# Hàm createSimpsonsModel được sử dụng để xây dựng một mô hình mạng nơ-ron sử dụng các thông số này và sau đó trả về mô hình đã được cấu hình sẵn để sử dụng trong quá trình huấn luyện.
#  Mô hình này sẽ được sử dụng để phân loại các nhân vật trong loạt phim hoạt hình Simpsons dựa trên dữ liệu huấn luyện và các tham số bạn đã cung cấ

# Đối tượng mô hình mạng nơ-ron là một thực thể (object) trong ngôn ngữ lập trình, được tạo ra và cấu hình để thực hiện các tác vụ liên quan đến mạng nơ-ron như phân loại ảnh,
#  dự đoán dữ liệu, hoặc giải quyết các bài toán khác dựa trên học máy và trí tuệ nhân tạo.

# Một mô hình mạng nơ-ron bao gồm các thành phần chính như lớp (layer), trọng số (weight), hàm kích hoạt (activation function), và các thông số cấu hình như tốc độ học (learning rate),
#  hàm mất mát (loss function), và thuật toán tối ưu hóa. Mô hình này có khả năng học từ dữ liệu huấn luyện và sau đó được sử dụng để dự đoán hoặc phân loại dữ liệu mới.

# Trong ngữ cảnh của học máy và deep learning, mô hình mạng nơ-ron thường là một đối tượng đại diện cho kiến thức và khả năng dự đoán của mạng nơ-ron. Đối với mạng nơ-ron sâu 
# (deep neural networks), mô hình có thể bao gồm nhiều lớp (hidden layers) và hàng loạt các trọng số, được điều chỉnh trong quá trình huấn luyện để tối ưu hóa hiệu suất trong
#  việc thực hiện các nhiệm vụ cụ thể.

# Tham số loss trong hàm tạo mô hình mạng nơ-ron được sử dụng để xác định hàm mất mát (loss function) mà mô hình sẽ sử dụng trong quá trình huấn luyện. Hàm mất mát là một phần 
# quan trọng của quá trình huấn luyện mạng nơ-ron, và nó định nghĩa cách tính sai số giữa đầu ra dự đoán của mô hình và giá trị thực tế trong tập dữ liệu huấn luyện.

# Trong ví dụ của bạn, bạn đặt loss là 'binary_crossentropy', cho biết rằng mô hình sẽ sử dụng hàm mất mát binary cross-entropy. Hàm này thường được sử dụng trong bài toán
#  phân loại nhị phân, nơi mô hình phải dự đoán giữa hai lớp (chẳng hạn, lớp tích cực và lớp tiêu cực). Hàm mất mát binary cross-entropy giúp đo lường sai số giữa đầu ra dự 
# đoán và giá trị thực tế bằng cách sử dụng hàm log-loss.

# Sự lựa chọn của hàm mất mát thường phụ thuộc vào loại bài toán bạn đang giải quyết. Có nhiều loại hàm mất mát khác nhau cho các loại bài toán khác nhau, và việc chọn một 
# hàm mất mát thích hợp là quan trọng để đảm bảo mô hình học tốt và đạt được hiệu suất cao.

# Tốc độ học (learning rate) là một siêu tham số quan trọng trong quá trình huấn luyện mạng nơ-ron. Nó quy định tốc độ cập nhật trọng số của mạng theo hướng để giảm thiểu hàm mất mát. 
# ốc độ học quyết định khoảng cách mà mô hình di chuyển trong không gian trọng số sau mỗi lần cập nhật. Nếu bạn đặt tốc độ học quá lớn, mô hình có thể bị dao động và không hội tụ. 
# Nếu bạn đặt tốc độ học quá nhỏ, thì quá trình học sẽ diễn ra rất chậm và có thể bị kẹt tại các điểm tối thiểu cục.

# Tốc độ học được chọn phụ thuộc vào bài toán cụ thể và kiến thức thực nghiệm. Các giá trị thông thường cho tốc độ học thường nằm trong khoảng từ 0.1 đến 0.0001, tuy nhiên, 
# không có một giá trị cố định. Quá trình tinh chỉnh tốc độ học có thể yêu cầu thử nghiệm nhiều giá trị khác nhau để xác định giá trị phù hợp.

# Nếu tốc độ học quá lớn, mô hình có thể hội tụ nhanh chóng, nhưng có thể không hội tụ đến một giải pháp tối ưu. Nếu tốc độ học quá nhỏ, quá trình học sẽ rất chậm và
#  có thể bị kẹt ở các giá trị tối ưu cục.

# Thường, người ta sử dụng kỹ thuật tinh chỉnh tốc độ học trong quá trình huấn luyện mô hình để tìm giá trị tốc độ học tốt nhất cho từng bài toán cụ thể.

# Thuật toán tối ưu hóa là một phần quan trọng trong quá trình huấn luyện mạng nơ-ron và trong nhiều bài toán tối ưu hóa khác. Nó đề cập đến việc tìm kiếm giải pháp 
# tốt nhất trong một không gian đối tượng mục tiêu, thường là một hàm mất mát hoặc hàm mục tiêu.

# Các thuật toán tối ưu hóa có nhiệm vụ điều chỉnh các tham số của mô hình (trọng số và thiên lệch) để giảm thiểu giá trị của hàm mất mát. Mục tiêu cuối cùng là tìm ra giá 
# trị của các tham số mà làm giảm thiểu hàm mất mát.

# Một số thuật toán tối ưu hóa phổ biến bao gồm:

# Gradient Descent (GD) và các biến thể của nó như Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent: Các thuật toán dựa trên gradient để điều chỉnh tham số mô 
# hình dựa trên đạo hàm của hàm mất mát.

# Adam: Một thuật toán tối ưu hóa kết hợp các lợi ích của các biến thể khác nhau của gradient descent. Nó thường được sử dụng rộng rãi trong mạng nơ-ron.

# RMSprop: Một thuật toán tối ưu hóa dựa trên gradient descent, chú trọng đến việc điều chỉnh tốc độ học cho từng tham số.

# L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno): Một thuật toán tối ưu hóa sử dụng kỹ thuật tạo ra một mô hình xấp xỉ của hàm mất mát.

# Conjugate Gradient: Một phương pháp tối ưu hóa sử dụng hướng tìm kiếm ẩn trong không gian của các tham số.
# NAG (Nesterov Accelerated Gradient), còn được gọi là "Hướng dẫn đà được gia tốc Nesterov," là một phương pháp tối ưu hóa trong học máy, thường được sử dụng trong quá trình huấn 
# luyện mạng nơ-ron. Phương pháp này là một biến thể của Gradient Descent được cải tiến bằng việc sử dụng "đà" (momentum) để tăng cường quá trình cập nhật trọng số.

# Cơ chế hoạt động của NAG như sau:

# Trong mỗi bước, NAG đầu tiên sử dụng đà (momentum) để tính toán sự thay đổi ước lượng của gradient theo trọng số tại vị trí dự kiến sau một bước tiến.
# Sau đó, nó sử dụng gradient của hàm mất mát tại vị trí dự kiến để điều chỉnh đà theo hướng tính toán từ bước 1.
# Cuối cùng, nó áp dụng cập nhật trọng số dựa trên đà mới được tính toán từ bước 2.
# NAG giúp tăng cường quá trình cập nhật trọng số bằng cách sử dụng thông tin từ vị trí dự kiến trước khi cập nhật. Điều này giúp tránh hiện tượng "quá băng" (overshooting),
#  tức là tránh việc đi quá xa so với điểm tối ưu cục bộ, giúp tăng tốc quá trình học.

# NAG thường là một biến thể hiệu quả của Gradient Descent và thường được sử dụng trong quá trình huấn luyện các mạng nơ-ron sâu.

# Một số thuật ngữ mà bạn đang gặp trong đoạn trích đó:

# Tốc độ học (Learning Rate): Đây là một tham số quan trọng trong quá trình huấn luyện mô hình máy học. Nó quy định độ lớn của các bước cập nhật trọng số trong quá trình 
# tối ưu hóa. Nếu tốc độ học quá lớn, các bước cập nhật trọng số có thể lớn đến mức mô hình không hội tụ, và nó có thể "dao động" xung quanh điểm tối ưu thay vì hội tụ đến nó.

# Điểm tối ưu: Đây là điểm trong không gian của các trọng số mà chúng ta cố gắng tối thiểu hóa hàm mất mát. Điểm tối ưu có thể là một điểm tối thiểu cục (local minimum)
#  hoặc điểm tối ưu toàn cục (global minimum) của hàm mất mát. Mục tiêu trong huấn luyện mô hình là tìm ra điểm tối ưu để mô hình có thể làm các dự đoán tốt nhất.

# Tối thiểu cục (Local Minimum): Đây là một điểm tối ưu trong không gian của các trọng số, nhưng nó chỉ là tối ưu trong một phạm vi cụ thể. Các local minimum có thể xuất 
# hiện trong hàm mất mát khi có nhiều điểm tối ưu hơn một. Mô hình có thể bị kẹt tại các local minimum nếu không đủ tốt trong việc thoát khỏi chúng.

# Cập nhật trọng số: Trong quá trình huấn luyện mô hình máy học, trọng số (weights) của mô hình được điều chỉnh bằng cách cập nhật chúng dựa trên đạo hàm của hàm mất mát. 
# Tốc độ học quy định độ lớn của các cập nhật trọng số.

# Khi tốc độ học quá lớn, các cập nhật trọng số có thể bị đi quá xa và tạo ra các dao động xung quanh điểm tối ưu, điều này làm cho mô hình không hội tụ (không đạt được 
# điểm tối ưu). Ngược lại, nếu tốc độ học quá nhỏ, các cập nhật trọng số rất nhỏ và quá trình học sẽ diễn ra rất chậm hoặc có thể bị kẹt tại các điểm tối thiểu cục thay 
# vì tiến đến điểm tối ưu toàn cục. Điều quan trọng là phải điều chỉnh tốc độ học để đạt được sự cân bằng giữa tốc độ học và sự hội tụ của mô hình.
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters),
                                         loss='binary_crossentropy', learning_rate=0.001, momentum=0.9,
                                         nesterov=True)

# model.summary() là một câu lệnh trong deep learning sử dụng để hiển thị tóm tắt của mô hình mạng nơ-ron, bao gồm thông tin về kiến trúc của mô hình, số lượng tham số, và 
# kích thước đầu ra của mỗi lớp.

# Kết quả đầu ra của model.summary() bao gồm:

# Layer (Lớp): Danh sách các lớp trong mô hình, theo thứ tự từ lớp đầu vào đến lớp đầu ra.
# Output Shape (Kích thước Đầu ra): Mô tả kích thước đầu ra của mỗi lớp. Kích thước này thường là một tuple, ví dụ (None, 64, 64, 32), trong đó "None" thường là kích thước batch 
# (số lượng mẫu được đưa vào cùng một lúc).
# Param # (Số tham số): Số lượng tham số trong mỗi lớp. Đây bao gồm số lượng trọng số (weights) và số lượng tham số điều chỉnh (biases).
# Connected to (Kết nối đến): Liệt kê các lớp được kết nối với lớp hiện tại.
# Sử dụng model.summary() là một cách thuận tiện để kiểm tra kiến trúc của mô hình, đảm bảo rằng nó được xây dựng đúng cách và để kiểm tra số lượng tham số, điều này quan trọng 
# để quản lý tài nguyên và đảm bảo mô hình có thể huấn luyện một cách hiệu quả.
model.summary()

# Training the model
# Hàm lr_schedule là một hàm để xác định tốc độ học (learning rate) của mô hình trong quá trình huấn luyện dựa trên số lượng epoch (vòng lặp huấn luyện). Hàm này 
# thường được sử dụng để giảm tốc độ học theo thời gian, giúp quá trình huấn luyện ổn định hơn và cải thiện khả năng hội tụ của mô hình.

# Trong trường hợp cụ thể này, hàm lr_schedule xác định tốc độ học ban đầu là 0.001 và sau mỗi epoch, nó sẽ nhân với 0.9. Điều này có nghĩa rằng sau mỗi epoch, 
# tốc độ học sẽ giảm đi 10% so với epoch trước. Điều này giúp mô hình học một cách dần dần và ổn định hơn.

# Sử dụng một tốc độ học giảm dần có lợi ích làm giảm nguy cơ mô hình bị dao động và tối ưu hóa tốt hơn trong quá trình huấn luyện. Điều này thường được áp dụng
#  trong các vấn đề phức tạp và khi tốc độ học ban đầu lớn có thể gây ra sự dao động trong quá trình học.
# Khả năng hội tụ của mô hình là khả năng của mô hình máy học hoặc mạng nơ-ron để học và tiến dần cải thiện hiệu suất dự đoán trên dữ liệu huấn luyện theo thời gian. 
# Quá trình hội tụ xuất hiện khi mô hình đã học đủ tốt để đạt được một hiệu suất dự đoán ổn định trên tập dữ liệu huấn luyện và không còn biến thiên lớn trong hiệu suất.

# Khả năng hội tụ là một yếu tố quan trọng trong quá trình huấn luyện mô hình. Một mô hình có khả năng hội tụ tốt sẽ học nhanh hơn và dễ dàng đạt được hiệu suất tốt trên 
# dữ liệu kiểm tra. Ngược lại, mô hình không có khả năng hội tụ có thể dẫn đến hiệu suất không ổn định và biến động lớn trên dữ liệu kiểm tra.

# Các yếu tố ảnh hưởng đến khả năng hội tụ của mô hình bao gồm tốc độ học, kiến trúc mạng, số lượng dữ liệu huấn luyện, và các siêu tham số khác. Điều quan trọng là điều 
# chỉnh các yếu tố này một cách hợp lý để đảm bảo mô hình hội tụ và đạt được hiệu suất tốt trên dữ liệu mới.
def lr_schedule(epoch):
    return 0.001 * 0.9 ** epoch

# Dòng code này tạo một danh sách các callbacks cho quá trình huấn luyện mô hình. Trong trường hợp này, bạn đang sử dụng một callback được gọi là LearningRateScheduler để
#  điều chỉnh tốc độ học (learning rate) của mô hình theo một lịch trình cụ thể.

# LearningRateScheduler là một callback trong TensorFlow/Keras cho phép bạn điều chỉnh tốc độ học (learning rate) của mô hình theo một lịch trình xác định. Lịch trình 
# này có thể dựa trên số epoch hoặc các yếu tố khác.

# Trong ví dụ này, bạn đã định nghĩa một hàm lr_schedule(epoch) để xác định tốc độ học dựa trên số epoch. Cụ thể, tốc độ học là kết quả của phép nhân 0.001 với 0.9 mũ
#  epoch, nghĩa là nó giảm dần theo số epoch. Điều này giúp trong quá trình huấn luyện, tốc độ học giảm dần theo thời gian, cho phép mô hình học tốt hơn và hội tụ một cách ổn định.

# Khi bạn đặt callback LearningRateScheduler(lr_schedule), mô hình sẽ sử dụng lịch trình này để điều chỉnh tốc độ học sau mỗi epoch, giúp quá trình huấn luyện được điều 
# chỉnh một cách tốt hơn để đạt được hiệu suất tốt hơn trên dữ liệu kiểm tra.
callbacks_list = [LearningRateScheduler(lr_schedule)]

# Hàm fit là một phần quan trọng trong quá trình huấn luyện mô hình học máy trong TensorFlow/Keras. Nó được sử dụng để thực hiện quá trình huấn luyện thực tế trên dữ liệu 
# huấn luyện và đánh giá hiệu suất của mô hình trên dữ liệu kiểm tra. Dưới đây là giải thích về các tham số được chuyền vào hàm fit:

# train_gen: Đây là generator (nguồn cung cấp dữ liệu) sẽ cung cấp dữ liệu huấn luyện trong quá trình huấn luyện mô hình. Generator này tạo ra dữ liệu từ x_train và 
# y_train bằng cách chia thành các batch nhỏ. Mỗi epoch, mô hình sẽ được huấn luyện trên từng batch này.

# steps_per_epoch: Đây là số lượng batch được sử dụng trong mỗi epoch. Nó thường được tính bằng cách chia tổng số mẫu huấn luyện cho kích thước batch (BATCH_SIZE).

# epochs: Số lượng epoch (vòng lặp huấn luyện) mà mô hình sẽ thực hiện. Một epoch tương đương với việc huấn luyện trên toàn bộ dữ liệu huấn luyện một lần. Thông thường, 
# bạn sẽ cần lặp qua nhiều epoch để mô hình học tốt.

# validation_data: Dữ liệu kiểm tra (x_val, y_val) được sử dụng để đánh giá hiệu suất của mô hình sau mỗi epoch. Mô hình sẽ không được huấn luyện trên dữ liệu kiểm tra, 
# nhưng sẽ được kiểm tra trên nó để tính các độ đo như accuracy, loss, v.v.

# validation_steps: Số lượng batch kiểm tra trong mỗi epoch. Tương tự như steps_per_epoch, thường được tính bằng cách chia tổng số mẫu kiểm tra cho kích thước batch (BATCH_SIZE).

# callbacks: Một danh sách các callbacks sẽ được gọi trong quá trình huấn luyện. Trong ví dụ của bạn, callbacks_list chứa một callback để điều chỉnh tốc độ học theo lịch trình.

# Hàm fit sẽ thực hiện quá trình đệ quy qua các epoch. Trong mỗi epoch, nó sẽ lặp qua các batch trong train_gen, thực hiện lan truyền tiến (forward pass) và lan truyền ngược
#  (backward pass) để điều chỉnh trọng số của mô hình. Sau mỗi epoch, nó sẽ sử dụng dữ liệu kiểm tra để đánh giá hiệu suất của mô hình và in ra các độ đo như accuracy, loss.
#  Quá trình này lặp lại cho đến khi số epoch đã định sẵn được hoàn thành.

# Trong kết quả trả về của hàm fit, bạn sẽ có thông tin về quá trình huấn luyện, bao gồm loss và accuracy trên dữ liệu huấn luyện và kiểm tra sau mỗi epoch.
# Quá trình "lan truyền tiến" và "lan truyền ngược" là phần quan trọng trong quá trình huấn luyện mô hình học máy.

# Lan truyền tiến (Forward Pass):

# Lan truyền tiến là quá trình mà đầu vào của mô hình (dữ liệu huấn luyện) được truyền qua mạng nơ-ron để tính toán đầu ra dự đoán.
# Dữ liệu được truyền từ lớp này sang lớp khác trong mạng nơ-ron.
# Mỗi lớp sẽ thực hiện một số phép toán để biến đổi dữ liệu và truyền nó đến lớp tiếp theo.
# Kết quả cuối cùng sau quá trình lan truyền tiến là dự đoán của mô hình về đầu vào.
# Lan truyền ngược (Backward Pass):

# Lan truyền ngược là quá trình sử dụng đầu ra dự đoán để tính toán độ lỗi (loss) của mô hình.
# Sau khi tính toán độ lỗi, quá trình lan truyền ngược sẽ lan truyền lỗi từ lớp cuối cùng trở về lớp đầu tiên của mạng nơ-ron để xác định cách điều chỉnh các trọng số của mô hình để làm giảm độ lỗi.
# Quá trình này thực hiện các phép toán đạo hàm để xác định hướng và mức độ điều chỉnh cần thiết cho từng trọng số của mô hình.
# Tóm lại, quá trình lan truyền tiến tính toán dự đoán của mô hình, trong khi quá trình lan truyền ngược tính toán độ lỗi và điều chỉnh các trọng số để cải thiện hiệu suất của mô hình. Hai quá trình này lặp lại trong mỗi epoch của quá trình huấn luyện để dần dần cải thiện mô hình.
training = model.fit(train_gen,
                    steps_per_epoch=len(x_train) // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_val, y_val),
                    validation_steps=len(y_val) // BATCH_SIZE,
                    callbacks=callbacks_list)

print(characters)

"""## Testing"""

test_path = r'E:\simpsons_dataset/charles_montgomery_burns_0.jpg'

img = cv.imread(test_path)

plt.imshow(img)
plt.show()
# Hàm prepare(image) được sử dụng để tiền xử lý một hình ảnh trước khi đưa nó vào mô hình để kiểm tra. Quá trình tiền xử lý là cần thiết để đảm bảo rằng hình ảnh đầu vào có cùng 
# định dạng và đặc trưng với dữ liệu huấn luyện. Trong trường hợp của hàm prepare(image) trong mã của bạn:

# cv.cvtColor(image, cv.COLOR_BGR2GRAY): Đây là bước chuyển đổi hình ảnh màu sang hình ảnh đen trắng (gray scale). Mô hình của bạn có thể được huấn luyện với dữ liệu hình ảnh đen 
# trắng nên bạn cần chuyển đổi hình ảnh màu thành đen trắng để đảm bảo rằng đầu vào có cùng định dạng với dữ liệu huấn luyện.

# cv.resize(image, IMG_SIZE): Ở đây, bạn thay đổi kích thước của hình ảnh để nó có kích thước (chiều cao và chiều rộng) là IMG_SIZE. Điều này cũng là để đảm bảo rằng tất cả hình ảnh
#  đều có kích thước thống nhất, giúp mô hình hoạt động hiệu quả.

# caer.reshape(image, IMG_SIZE, 1): Cuối cùng, bạn sử dụng caer.reshape để chuyển đổi hình ảnh thành một mảng numpy với kích thước là IMG_SIZE và một kênh (channel). Kênh là 1 vì bạn
#  đã chuyển hình ảnh thành đen trắng. Điều này làm cho hình ảnh có cùng định dạng với dữ liệu huấn luyện.

# Hàm prepare(image) giúp bạn đảm bảo rằng hình ảnh kiểm tra có định dạng phù hợp và sẵn sàng để đưa vào mô hình để thực hiện dự đoán.
def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image

predictions = model.predict(np.array([prepare(img)]))

# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])
