#include <iostream>
#include <opencv2/opencv.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

using namespace std;
using namespace cv;

// 辅助函数：将 matd_t* 转换为 cv::Mat
cv::Mat matd_to_cvmat(matd_t* mat) {
    cv::Mat cvmat(mat->nrows, mat->ncols, CV_64F);
    for (int i = 0; i < mat->nrows; i++) {
        for (int j = 0; j < mat->ncols; j++) {
            cvmat.at<double>(i, j) = mat->data[i * mat->ncols + j];
        }
    }
    return cvmat;
}

int main() {
    // 网络初始化
    int sock = 0;
    struct sockaddr_in serv_addr;
    const char* host_ip = "192.168.127.166";  // 我的电脑IP
    const int port = 5000;//端口

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        cerr << "创建失败" << endl;
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host_ip, &serv_addr.sin_addr) <= 0) {
        cerr << "无效IP" << endl;
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        cerr << "连接失败" << endl;
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "相机有问题" << endl;
        return -1;
    }
//分辨率
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);

    td->quad_decimate = 2.0;
    td->quad_sigma = 0.0;
    td->nthreads = 4;

    const double TAG_SIZE_M = 0.1;
    double fx = 500.0, fy = 500.0, cx = 0.0, cy = 0.0;

    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cx = frame.cols / 2.0;
        cy = frame.rows / 2.0;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        image_u8_t im = {gray.cols, gray.rows, gray.cols, gray.data};
        zarray_t *detections = apriltag_detector_detect(td, &im);

        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            apriltag_detection_info_t info = {
                .det = det,
                .tagsize = TAG_SIZE_M,
                .fx = fx, .fy = fy,
                .cx = cx, .cy = cy
            };

            apriltag_pose_t pose;
            estimate_tag_pose(&info, &pose);

            for (int j = 0; j < 4; j++) {
                line(frame,
                     Point(det->p[j][0], det->p[j][1]),
                     Point(det->p[(j+1)%4][0], det->p[(j+1)%4][1]),
                     Scalar(0, 255, 0), 2);
            }

            Point2f center(det->c[0], det->c[1]);
            string id_text = to_string(det->id);
            putText(frame, id_text,
                    Point(center.x - 20, center.y + 20),
                    FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 255, 0), 2);

            matd_destroy(pose.R);
            matd_destroy(pose.t);
        }

        // 红点追踪，但是只能追踪
        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Scalar lower_red1(0, 100, 100), upper_red1(10, 255, 255);
        Scalar lower_red2(160, 100, 100), upper_red2(179, 255, 255);
        Mat mask1, mask2;
        inRange(hsv, lower_red1, upper_red1, mask1);
        inRange(hsv, lower_red2, upper_red2, mask2);
        mask = mask1 | mask2;

//去噪
        erode(mask, mask, Mat(), Point(-1, -1), 2);
        dilate(mask, mask, Mat(), Point(-1, -1), 2);
//轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            double max_area = 0;
            int max_idx = 0;
            for (int i = 0; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    max_idx = i;
                }
            }

            if (max_area > 5) {
                Moments M = moments(contours[max_idx]);
                int cx_laser = int(M.m10 / M.m00);
                int cy_laser = int(M.m01 / M.m00);
//标点
                circle(frame, Point(cx_laser, cy_laser), 5, Scalar(255, 0, 255), -1);
                line(frame, Point(cx_laser, 0), Point(cx_laser, frame.rows), Scalar(255, 0, 255), 1);
                line(frame, Point(0, cy_laser), Point(frame.cols, cy_laser), Scalar(255, 0, 255), 1);

                int vertical_offset = cy_laser - frame.rows / 2;
               
            }
        }
        // 结束

        Point frame_center(frame.cols/2, frame.rows/2);
        int cross_len = 15;
        line(frame, Point(frame_center.x - cross_len, frame_center.y),
                Point(frame_center.x + cross_len, frame_center.y), Scalar(0, 0, 255), 2);
        line(frame, Point(frame_center.x, frame_center.y - cross_len),
                Point(frame_center.x, frame_center.y + cross_len), Scalar(0, 0, 255), 2);

       //传到电脑
        vector<uchar> buf;
        imencode(".jpg", frame, buf, {IMWRITE_JPEG_QUALITY, 80});

        try {
            uint32_t img_size = htonl(buf.size());
            send(sock, &img_size, sizeof(img_size), 0);
            send(sock, buf.data(), buf.size(), 0);
        } catch (const exception& e) {
            cerr << "传输失败" << e.what() << endl;
        }

        apriltag_detections_destroy(detections);
    }

    close(sock);
    cap.release();
    destroyAllWindows();
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    return 0;
}
