#include "me_header.h"

int main(int argc, char *argv[])
{
	for (int now_loop = 1; now_loop <= PROJECT_LOOP; now_loop++) {
		// read_file, write_fileをループごとに適切な名前に変更する
		std::string read_file = basis_read_file;
		std::string write_file = basis_write_file;

		change_filename(read_file, write_file, now_loop);

		log_write(read_file, write_file);
		str_checker(read_file, write_file);

		if (!overwrite_check(write_file)) {
			continue;
		}

		cv::VideoCapture cap;
		cv::VideoWriter writer;
		std::vector<char> embed;
		std::ofstream ofs;
		cv::TickMeter meter;
		int framenum;
		cv::Size size;
		

		//初期化
		init_me(&cap, &embed, &size, &ofs, &writer, read_file, write_file, num_embedframe);
		int total_frames = cap.get(CV_CAP_PROP_FRAME_COUNT) - 1;  // 0フレーム分をカット

		do {
			int i;
			std::vector<cv::Mat> planes(3), planes2(3);  // 一時的に，YCrCbを格納する(同一フレームのYCrCbかBGRの3要素を格納している)
			std::vector<cv::Mat> luminance;               // 輝度値を一定のフレーム分格納する
			std::vector<cv::Mat> dst_luminance;           // 埋め込みを行ったフレーム分の輝度値を格納する
			std::vector<cv::Mat> Cr;                      
			std::vector<cv::Mat> Cb;
			cv::Mat frame_BGR(size, CV_8UC3), frame_BGR2(size, CV_8UC3);
			cv::Mat frame_YCrCb(size, CV_8UC3), frame_YCrCb2(size, CV_8UC3);
			std::vector<cv::Mat> check_motion_array;

			//timer startf
			meter.start();

			//preprocessing
			for (i = 0; i < num_embedframe; i++) {
				cap >> frame_BGR;
				if (frame_BGR.empty())
					break;

				//RGB画像をYCrCb画像に変換
				cv::cvtColor(frame_BGR, frame_YCrCb, CV_BGR2YCrCb);

				//チャンネルごとに分解
				cv::split(frame_YCrCb, planes);
				luminance.push_back(planes[0]);
				Cr.push_back(planes[1]);
				Cb.push_back(planes[2]);
				planes.clear();
			}

			//埋め込み処理(num_embeddframe分だけ処理を行う)
			motion_embedder(luminance, dst_luminance, embed, cap.get(CV_CAP_PROP_POS_FRAMES), num_embedframe, delta);

			//timer end
			meter.stop();

			//double psnr[num_embedframe];
			for (i = 0; i < num_embedframe; i++) {
				planes.push_back(dst_luminance[i]);
				planes.push_back(Cr[i]);
				planes.push_back(Cb[i]);

				//分解したチャンネルを合成
				cv::merge(planes, frame_YCrCb);

				//YCrCb画像をBGR画像へ変換
				cv::cvtColor(frame_YCrCb, frame_BGR, CV_YCrCb2BGR);

				//ファイル書き込み
				writer << frame_BGR;
				planes.clear();
			}

			std::cout << "frame" << cap.get(CV_CAP_PROP_POS_FRAMES) << std::endl;
			meter.reset();
			meter.start();
		} while (cap.get(CV_CAP_PROP_POS_FRAMES) < total_frames);

		//後処理
		cap.release();
		writer.release();
	}
	return 0;
}