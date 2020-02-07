#include "me_header.h"

int main(int argc, char *argv[])
{
	for (int now_loop = 1; now_loop <= PROJECT_LOOP; now_loop++) {
		// read_file, write_file�����[�v���ƂɓK�؂Ȗ��O�ɕύX����
		std::string read_file = basis_read_file;
		std::string write_file = basis_write_file;
		std::string motion_vector_file = basis_motion_vector_file;

		change_filename(read_file, write_file, motion_vector_file , now_loop);

		log_write(read_file, write_file);
		str_checker(read_file, write_file);

		//if (!overwrite_check(write_file)) {
		//	continue;
		//}

		cv::VideoCapture cap;
		cv::VideoWriter writer;
		std::vector<char> embed;
		std::ofstream ofs;
		cv::TickMeter meter;
		int framenum;
		cv::Size size;



		//������
		init_me(&cap, &embed, &size, &ofs, &writer, read_file, write_file, motion_vector_file, num_embedframe);
		int total_frames = cap.get(CV_CAP_PROP_FRAME_COUNT) - 1;  // 0�t���[�������J�b�g

		do {
			int i;
			std::vector<cv::Mat> planes;   // �c������8��f�����炷
			std::vector<cv::Mat> planes2;  // �ꎞ�I�ɁCYCrCb���i�[����(����t���[����YCrCb��BGR��3�v�f���i�[���Ă���)
			std::vector<cv::Mat> luminance;               // �P�x�l�����̃t���[�����i�[����
			std::vector<cv::Mat> dst_luminance;           // ���ߍ��݂��s�����t���[�����̋P�x�l���i�[����
			std::vector<cv::Mat> Cr;
			std::vector<cv::Mat> Cb;
			cv::Mat frame_BGR(size, CV_8UC3), frame_BGR2(size, CV_8UC3);
			cv::Mat frame_YCrCb(size, CV_8UC3), frame_YCrCb2(size, CV_8UC3);
			std::vector<cv::Mat> temp_lumi;
			std::vector<cv::Mat> temp_Cr;
			std::vector<cv::Mat> temp_Cb;

			
			
			
			//300f���ɕʂ̓��摜�Ƃ��ĕۑ�
			if (cap.get(CV_CAP_PROP_POS_FRAMES) == 300) {
				writer.release();
				if (read_file.find("avi") != std::string::npos) {
					writer = writer_open(write_file + "_2.avi", cap);
				}
			}
			if (cap.get(CV_CAP_PROP_POS_FRAMES) == 600) {
				writer.release();
				if (read_file.find("avi") != std::string::npos) {
					writer = writer_open(write_file + "_3.avi", cap);
				}
			}



			//preprocessing
			for (i = 0; i < num_embedframe; i++) {
				// 2020 add
				// planes initilize
				for (int j = 0; j < 3; j++) {
					planes.push_back(cv::Mat::zeros(cv::Size(FRAME_width, FRAME_height - 8), CV_8UC1));
					planes2.push_back(cv::Mat::zeros(cv::Size(FRAME_width, FRAME_height), CV_8UC1));
				}


				cap >> frame_BGR;
				if (frame_BGR.empty())
					break;

				//RGB�摜��YCrCb�摜�ɕϊ�
				cv::cvtColor(frame_BGR, frame_YCrCb, CV_BGR2YCrCb);

				//�`�����l�����Ƃɕ���
				cv::split(frame_YCrCb, planes2);

				for (int j = 0; j < 3; j++) {
					for (int y = 0; y < FRAME_height; y++) {
						for (int x = 0; x < FRAME_width; x++) {
							if (y < 16 * 67) {   // ���܂�̉�f�ȊO���i�[
								planes[j].at<unsigned char>(y, x) = planes2[j].at<unsigned char>(y, x);
							}
						}
					}
				}
				
				temp_lumi.push_back(planes2[0]);
				temp_Cr.push_back(planes2[1]);
				temp_Cb.push_back(planes2[2]);
				luminance.push_back(planes[0]);
				Cr.push_back(planes[1]);
				Cb.push_back(planes[2]);
				planes2.clear();
				planes.clear();
			}

			std::vector<mv_class> mv_all(num_embedframe);    

			//���ߍ��ݏ���(num_embeddframe�������������s��)
			motion_embedder(luminance, dst_luminance, embed, cap.get(CV_CAP_PROP_POS_FRAMES) - num_embedframe, num_embedframe, delta,motion_vector_file, mv_all);



			//double psnr[num_embedframe];
			for (i = 0; i < num_embedframe; i++) {

				// planes2 initilize
				for (int j = 0; j < 3; j++) {
					planes2.push_back(cv::Mat::zeros(cv::Size(FRAME_width, FRAME_height), CV_8UC1));
				}

				planes.push_back(dst_luminance[i]);
				planes.push_back(Cr[i]);
				planes.push_back(Cb[i]);



				for (int j = 0; j < 3; j++) {
					for (int y = 0; y < FRAME_height; y++) {
						for (int x = 0; x < FRAME_width; x++) {

							if (y < 16 * 67) {
								planes2[j].at<unsigned char>(y, x) = planes[j].at<unsigned char>(y, x);
							}
							else {
								if (j==0) {
									planes2[j].at<unsigned char>(y, x) = temp_lumi[i].at<uchar>(y, x);
								}
								else if (j == 1) {
									planes2[j].at<unsigned char>(y, x) = temp_Cr[i].at<uchar>(y, x);
								}
								else {
									planes2[j].at<unsigned char>(y, x) = temp_Cb[i].at<uchar>(y, x);
								}
							}
						}
					}
				}
				
	


				//���������`�����l��������
				cv::merge(planes2, frame_YCrCb);

				//YCrCb�摜��BGR�摜�֕ϊ�
				cv::cvtColor(frame_YCrCb, frame_BGR, CV_YCrCb2BGR);

				//�t�@�C����������
				writer << frame_BGR;
				planes.clear();
				planes2.clear();
			}

			std::cout << "frame" << cap.get(CV_CAP_PROP_POS_FRAMES) << std::endl;

			std::vector<mv_class>().swap(mv_all);

		} while (cap.get(CV_CAP_PROP_POS_FRAMES) < total_frames);

		//�㏈��
		cap.release();
		writer.release();
	}
	return 0;
}