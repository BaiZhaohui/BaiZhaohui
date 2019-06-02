# 图像编程入门
这样生成的图像由无符号字节（unsigned byte，C++中为unsigned char）构成，在OpenCV中用常量CV_8U 表示。另外，即使图像是作为灰度图像保存的，有时仍需要在读入时把它转换成三通道彩色图像。要实现这个功能，可把imread 函数的第二个参数设置为正数：
~~~C++
// 读取图像，并将其转换为三通道彩色图像
image= cv::imread("puppy.bmp", CV::IMREAD_COLOR);
~~~
这样创建的图像中，每个像素有3个字节，OpenCV中用CV_8UC3表示。如果输入的图像是灰度图像，则三个通道的值相同。

如果要在读入的图像时采用文件本身的格式，只需把第二个参数设置为负数。

可用channels方法检查图像的通道数：
~~~C++
cout << "This image has " << image.channels() << " channel(s)." << endl;
~~~
请注意，当用imread 打开路径指定不完整的图像时（前面例子的做法），imread 会自动采用默认目录。如果从控制台运行程序，默认目录显然就是当前控制台的目录；但是如果直接在IDE中运行程序，默认目录通常就是项目文件所在的目录。因此，要确保图像文件在正确的目录下。

当你用imshow 显示由整数（CV_16U 表示16 位无符号整数，CV_32S 表示32 位有符号整数）构成的图像时，图像每个像素的值会被除以256，以便能够在256 级灰度中显示。同样，在显示由浮点数构成的图像时，值的范围会被假设为0.0（显示黑色）~1.0（显示白色）。超出这个范围的值会显示为白色（大于1.0 的值）或黑色（小于0.0 的值）。

~~~C++
cv::flip(image,image,1); // 就地处理
~~~


### 关于Mat

cv::Mat 有两个必不可少的组成部分：一个头部和一个数据块。
头部包含了矩阵的所有相关信息（大小、通道数量、数据类型等），数据块包含了图像中所有像素的值。头部有一个指向数据块的指针，即data 属性。
cv::Mat 有一个很重要的属性，即只有在明确要求时，内存块才会被复制。实际上，大多数操作仅仅复制了cv::Mat 的头部，因此多个对象会指向同一个数据块。这种内存管理模式可以提高应用程序的运行效率，避免内存泄漏，但是我们必须了解它带来的后果。
新创建的cv::Mat 对象默认大小为0，但也可以指定一个初始大小，例如：
// 创建一个240 行×320 列的新图像
cv::Mat image1(240,320,CV_8U,100);

我们需要指定每个矩阵元素的类型，这里用CV_8U 表示每个像素对应1 字节（灰度图像），用字母U 表示无符号；你也可用字母S 表示有符号。对于彩色图像，你应该用三通道类型（CV_8UC3），也可以定义16 位和32 位的整数（有符号或无符号），例如CV_16SC3。我们甚至可以使用32 位和64 位的浮点数（例如CV_32F）。

图像（或矩阵）的每个元素都可以包含多个值（例如彩色图像中的三个通道），因此OpenCV引入了一个简单的数据结构cv::Scalar，用于在调用函数时传递像素值。该结构通常包含一个或三个值。如果要创建一个彩色图像并用红色像素初始化，可用如下代码：
~~~C++
// 创建一个红色图像
// 通道次序是BGR
cv::Mat(240,320,CV_8UC3,cv::Scalar(0,0,255))
//初始化灰度图像
cv::Mat image2(240,320,CV_8UC3,cv::Scalar(100));
~~~

可以随时用create 方法分配或重新分配图像的数据块。如果图像已被分配，其原来的内容会先被释放。出于对性能的考虑，如果新的尺寸和类型与原来的相同，就不会重新分配内存：
~~~C++
// 重新分配一个新图像
//（仅在大小或类型不同时）
image1.create(200,200,CV_8U);
~~~
一旦没有了指向cv::Mat 对象的引用，分配的内存就会被自动释放。这一点非常方便，因为它避免了C++动态内存分配中经常发生的内存泄漏问题。这是OpenCV（从第2 版开始引入）中的一个关键机制，它的实现方法是通过cv::Mat 实现计数引用和浅复制。因此，当在两幅图像之间赋值时，图像数据（即像素）并不会被复制，此时两幅图像都指向同一个内存块。这同样适用于图像间的值传递或值返回。由于维护了一个引用计数器，因此只有当图像的所有引用都将释放或赋值给另一幅图像时，内存才会被释放：
~~~C++
// 所有图像都指向同一个数据块
cv::Mat image4(image3);
image1= image3;
~~~
对上面图像中的任何一个进行转换都会影响到其他图像。如果要对图像内容做一个深复制，你可以使用copyTo 方法，目标图像将会调用create 方法。另一个生成图像副本的方法是clone，即创建一个完全相同的新图像：
~~~C++
// 这些图像是原始图像的新副本
image3.copyTo(image2);
cv::Mat image5= image3.clone();
~~~
如果需要把一幅图像复制到另一幅图像中，且两者的数据类型不一定相同，那就要使用convertTo方法了：
~~~C++
// 转换成浮点型图像[0,1]
image1.convertTo(image2,CV_32F,1/255.0,0.0)
//本例中的原始图像被复制进了一幅浮点型图像。这一方法包含两个可选参数：缩放比例和偏移量。需要注意的是，这两幅图像的通道数量必须相同。
~~~
cv::Mat 对象的分配模型还能让程序员安全地编写返回一幅图像的函数（或类方法）：
~~~C++
cv::Mat function() {
// 创建图像
cv::Mat ima(240,320,CV_8U,cv::Scalar(100));
// 返回图像
return ima;
}
~~~
我们还可以从main 函数中调用这个函数：
~~~C++
// 得到一个灰度图像
cv::Mat gray= function();
~~~
运行这条语句后，就可以用变量gray 操作这个由function 函数创建的图像，而不需要额外分配内存了。正如前面解释的，从cv::Mat 实例到灰度图像实际上只是进行了一次浅复制。当局部变量ima 超出作用范围后，ima 会被释放。但是从相关引用计数器可以看出，另一个实例（即变量gray）引用了ima 内部的图像数据，因此ima 的内存块不会被释放。
请注意，在使用类的时候要特别小心，不要返回图像的类属性。下面的实现方法很容易引发错误：
~~~C++
class Test {
// 图像属性
cv::Mat ima;
public:
// 在构造函数中创建一幅灰度图像
Test() : ima(240,320,CV_8U,cv::Scalar(100)) {}
// 用这种方法回送一个类属性，这是一种不好的做法
cv::Mat method() { return ima; }
};
~~~
如果某个函数调用了这个类的method，就会对图像属性进行一次浅复制。副本一旦被修改，class 属性也会被“偷偷地”修改，这会影响这个类的后续行为（反之亦然）。这违反了面向对象编程中重要的封装性原理。为了避免这种类型的错误，你需要将其改成返回属性的一个副本。
#### 扩展
1. 输入和输出数组
在OpenCV 的文档中，很多方法和函数都使用cv::InputArray 类型作为输入参数。cv::InputArray 类型是一个简单的代理类，用来概括OpenCV 中数组的概念，避免同一个方法或函数因为使用了不同类型的输入参数而有多个版本。也就是说，你可以在参数中使用cv::Mat对象或者其他的兼容类型。因为它是一个输入数组，所以你必须确保函数不会修改这个数据结构。有趣的是，cv::InputArray 也能使用常见的std::vector 类来构造；也就是说，用这种方式构造的对象可以作为OpenCV 方法和函数的输入参数（但千万不要在自定义类和函数中使用这个类）。其他兼容的类型有cv::Scalar 和cv::Vec，后者将在下一章介绍。此外还有一个代理类cv::OutputArray，用来指定某些方法或函数的返回数组。
2. 处理小矩阵
使用模板类cv::Matx和它的子类。
~~~C++
// // 3×3 双精度型矩阵
cv::Matx33d matrix(3.0, 2.0, 1.0,
2.0, 1.0, 3.0,
1.0, 2.0, 3.0);
// 3×1 矩阵（即向量）
cv::Matx31d vector(5.0, 1.0, 3.0);
// 相乘
cv::Matx31d result = matrix*vector;
//这些矩阵可以进行常见的数学运算。
~~~

### 定义ROI

ROI 实际上就是一个cv::Mat 对象，它与它的父图像指向同一个数据缓冲区，并且在头部指明了ROI 的坐标。接着，可以用下面的方法插入标志：
~~~C++
// 在图像的右下角定义一个ROI
cv::Mat imageROI(image,
      cv::Rect(image.cols-logo.cols, // ROI 坐标
                  image.rows-logo.rows,
                  logo.cols,logo.rows)); // ROI 大小
// 插入标志
logo.copyTo(imageROI);
~~~
这里的image 是目标图像，logo 是标志图像（相对较小）。
ROI 还可以用行和列的值域来描述。值域是一个从开始索引到结束索引的连续序列（不含开始值和结束值），可以用cv::Range 结构来表示这个概念。因此，一个ROI 可以用两个值域来定义。本例中的ROI 也可以定义为：
~~~C++
imageROI = image(cv::Range(image.rows - logo.rows,image.rows),
                              cv::Range(image.cols - logo.cols,image.cols));
~~~

cv::Mat 的operator()函数返回另一个cv::Mat 实例，可供后续使用。由于图像和ROI共享了同一块图像数据，因此ROI 的任何转变都会影响原始图像的相关区域。在定义ROI 时，数据并没有被复制，因此它的执行时间是固定的，不受ROI 尺寸的影响。要定义由图像中的一些行组成的ROI，可用下面的代码：
~~~C++
cv::Mat imageROI= image.rowRange(start,end);
~~~
与之类似，要定义由图像中一些列组成的ROI，可用下面的代码：
~~~C++
cv::Mat imageROI= image.colRange(start,end);
~~~
#### 扩展
**使用图像掩码**
函数或方法通常对图像中所有的像素进行操作，通过定义掩码可以限制这些函数或方法的作用范围。
掩码是一个8位图像，如果掩码中某个位置的值不为0，在这个位置上的操作就会起作用；如果掩码中某些像素位置的值为0，那么对图像中相应位置的操作将不起作用。
~~~C++
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
int main()
{
	cv::namedWindow("Image");
	cv::Mat image = cv::imread("E:\\image\\test\\5.jpg");
	cv::Mat logo = cv::imread("E:\\image\\images\\smalllogo.png");
	//定义ROI，在image的右下角
	cv::Mat imageROI(image,
		cv::Rect(image.cols - logo.cols,
			image.rows - logo.rows,
			logo.cols, logo.rows));

	//插入logo
	logo.copyTo(imageROI);
	cv::imshow("Image", image);
	cv::waitKey(0);
	//重新载入image
	image = cv::imread("E:\\image\\test\\5.jpg");
	//定义ROI，在image的右下角
	imageROI = image(cv::Rect(image.cols - logo.cols, image.rows - logo.rows, logo.cols, logo.rows));
	//或者用值域
	//imageROI = image(cv::Range(image.cols - logo.col, image.cols),cv::Range(image.rows - logo.rows, image.rows));

	//将logo用作掩码——必须为灰度图像
	cv::Mat mask(logo);
	//插入标志，只复制掩码不为0 的位置
	logo.copyTo(imageROI, mask);
	cv::imshow("Image", image);
	cv::waitKey(0);
	return 0;
}
~~~
