// 前のプログラムはtemp01のリポジトリに入れてます


#include "me_header.h"

int delta_thisfile = 0;
float average_thisfile = 0;


#define dx 0
#define dy 1

static double cosine_table[block_height][block_width];  // DCT変換用のコサインテーブル
static int matrix_height = 134;
static int matrix_width = 120;

#define MV_DETA_SIZE CV_8SC1
#define NP_DETA_SIZE CV_16SC1

#define MV_DETA_TYPE char
#define NP_DETA_TYPE short


void init_me(cv::VideoCapture* cap, std::vector<char>* embed, cv::Size* size, std::ofstream* ofs, cv::VideoWriter* writer, std::string read_file, std::string write_file, std::string motion_vector_file, int num_embedframe) {
	*embed = set_embeddata(embed_file);
	*cap = capture_open(read_file);
	//	*writer = mp4_writer_open(write_file + ".mp4", *cap);  // mp4なのでデータ量が小さいため分割の必要はない．．
	*writer = writer_open(write_file + "_1.avi", *cap);
	size->width = cap->get(CV_CAP_PROP_FRAME_WIDTH);
	size->height = cap->get(CV_CAP_PROP_FRAME_HEIGHT);
}

void set_ctable() {    //DCT変換で使うテーブルを初期設定
	std::ifstream ifs;
	ifs.open(cosine_file);
	if (ifs.fail()) {
		std::cerr << "can't open embed data file\n";
		std::getchar();
		exit(3);
	}

	for (int i = 0; i < block_height; i++) {
		for (int j = 0; j < block_width; j++) {
			cosine_table[i][j] = ifs.get();
		}
	}
}

std::vector<char> set_embeddata(const std::string filename) {
	std::vector<char> embed;
	std::ifstream ifs;
	ifs.open(filename);
	if (ifs.fail()) {
		std::cerr << "can't open embed data file\n";
		std::getchar();
		exit(3);
	}

	while (!ifs.eof())
		embed.push_back(ifs.get());

	return embed;
}

cv::VideoCapture capture_open(const std::string read_file) {
	cv::VideoCapture cap(read_file);
	if (!cap.isOpened()) {
		std::cout << "can't open video file.\n";
		std::getchar();
		exit(4);
	}
	return cap;
}

cv::VideoWriter writer_open(const std::string write_file, cv::VideoCapture cap) {
	cv::VideoWriter writer;
	cv::Size size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	writer.open(write_file, CV_FOURCC('D', 'I', 'B', ' '), cap.get(CV_CAP_PROP_FPS), size);
	if (!writer.isOpened())
		exit(5);
	return writer;
}

cv::VideoWriter mp4_writer_open(const std::string write_file, cv::VideoCapture cap) {
	cv::VideoWriter writer;
	cv::Size size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	writer.open(write_file, CV_FOURCC('M', 'P', '4', 'V'), cap.get(CV_CAP_PROP_FPS), size);
	if (!writer.isOpened())
		exit(5);
	return writer;
}

void set_motionvector(const std::string motion_vector_file, std::vector<mv_class>& mv_all, int cframe) {
	// ファイル内のデータが膨大すぎるため、埋め込み時に適宜この関数を呼び出して、その都度ファイルからデータを読み出す
	// 埋め込む先頭フレーム(cframe)までファイル読み込みを飛ばしてからnum_embedframe分データを取得する

	std::ifstream ifs;
	ifs.open(motion_vector_file);
	if (ifs.fail()) {
		std::cerr << "can't open motion vector data file\n";
		std::getchar();
		exit(3);
	}

	std::string str;
	std::string cframe_str = std::to_string(cframe);
	cframe_str.insert(0, "pts=");
	while (str.find(cframe_str) == std::string::npos) {   //pts=cframeとなるまで飛ばす
		getline(ifs, str, ' ');
	}


	mv_class temp_class;
	// 初期化
	temp_class.frame_index = -1;
	temp_class.x_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);
	temp_class.y_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);

	std::vector<int> debug_array(120);// debug用

	for (int pts = 1; pts < num_embedframe + 1 && !ifs.eof();) {
		int temp_start;
		std::string temp_str;
		int temp_count = 0;


		getline(ifs, str, ' ');


		// debug
		if (pts == 9) {
			int debug = 0;

			debug = 4;
		}

		// debug


		if (str.find("pts") != std::string::npos) {          // ptsが出れば、1フレームとみなす 
			pts++;
		}
		else if (str.find("frame_index") != std::string::npos) {
			temp_start = str.find("frame_index");
			temp_str = str.substr(temp_start + 12, str.length());        //動きベクトルファイル内では、frame_index=数字になっているという前提
			if (atoi(temp_str.c_str()) == -1) {
				temp_class.frame_index = atoi(temp_str.c_str());
			}
			else {
				temp_class.frame_index = pts - 1 + cframe;
			}
		}
		else if (str.find("shape") != std::string::npos) {
			temp_start = str.find("shape");

			for (int i = 0; i < 120 * matrix_height; i++) {
				// debug 
				if (i == 120 * 76) {
					int debug;
					debug = 77;
				}
				// debug


				getline(ifs, str, '\t');

				if (i == 0) {
					temp_str = str.substr(14, str.length() - 14);             //この行大丈夫かな・・・・
				}
				else if (i % 120 == 0) {   // 行の切れ目の\nを削除
					temp_str = str.substr(1, str.length() - 1);              // この行ダイジョブかな・・
				}
				else {
					temp_str = str;
				}

				if (i < 120 * matrix_height / 2) {
					temp_class.x_vector.at<MV_DETA_TYPE>(i / 120, i % 120) = atoi(temp_str.c_str());
				}
				else {
					temp_class.y_vector.at<MV_DETA_TYPE>((i / 2) / 120, (i / 2) % 120) = atoi(temp_str.c_str());
				}
			}


			// 深いコピー
			mv_all[pts - 1].frame_index = temp_class.frame_index;
			mv_all[pts - 1].x_vector = temp_class.x_vector.clone();
			mv_all[pts - 1].y_vector = temp_class.y_vector.clone();


			// 再度初期化
			temp_class.frame_index = -1;
			temp_class.x_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);
			temp_class.y_vector = cv::Mat::zeros(cv::Size(FRAME_width / motionvector_block_size, FRAME_height / motionvector_block_size), MV_DETA_SIZE);
		}
	}
}

cv::Mat filter(cv::Mat luminance) {                           // ブロック内の輝度値をならす(輝度値の平均化)
	cv::Mat dst_luminance(luminance.size(), CV_32F);
	int x = 0, y = 0, m = 0, n = 0;
	for (m = 0; m < luminance.cols; m += block_width) {        //  blockごとに処理をする
		for (n = 0; n < luminance.rows; n += block_height) {
			float mean = 0, sum = 0;
			for (x = m; x < m + block_width; x++) {             // ブロック内の合計輝度値を求める
				for (y = n; y < n + block_height; y++) {
					sum += (float)luminance.at<uchar>(y, x);
				}
			}

			mean = sum / (block_width * block_height);        // 上記の合計輝度値の平均値
			for (x = m; x < m + block_width; x++) {
				for (y = n; y < n + block_height; y++) {
					dst_luminance.at<float>(y, x) = mean;
				}
			}
		}
	}
	return dst_luminance;
}

float median(std::vector<float> v) {     // 中央値を返す
	int size = v.size();
	std::sort(v.begin(), v.end());

	if (size % 2 == 1) {
		return v[(size - 1) / 2];
	}
	else {
		return (v[(size / 2) - 1] + v[size / 2]) / 2;
	}
}

void motion_embedder(std::vector<cv::Mat>& luminance, std::vector<cv::Mat> &dst_luminance, std::vector<char> embed, int cframe, int num_embedframe, int delta, std::string motion_vector_file, std::vector<mv_class>& mv_all) {
	// cframeは20フレームの先頭フレーム番号が格納されている
	
	
	std::vector<cv::Mat> means;  //ブロック単位の平均輝度値を保持
	std::vector<cv::Mat> deviations;  //ブロック単位の平均値からの偏差を保持
	cv::Mat m_means = cv::Mat::zeros(FRAME_width, FRAME_height, CV_32F);  //mフレーム間での「ブロック単位の平均値」の平均値を保持
	std::vector<cv::Mat> m_deviations;  //mフレーム間での「ブロック単位の平均値」の平均値からの偏差を保持

	// ファイルからデータ取得
	set_motionvector(motion_vector_file, mv_all, cframe);

	// means, m_means , deviations, variance の作成
	for (int i = 0; i < num_embedframe; i++) {    //  各画素の偏差とブロック内平均輝度値を求める
		cv::Mat temp_luminance = luminance[i].clone();
		means.push_back(filter(temp_luminance));
		temp_luminance.convertTo(temp_luminance, CV_32F);
		deviations.push_back(temp_luminance - means[i]);      // 各画素の偏差 = 各画素の輝度値-ブロック内平均輝度値 
	}

	//mフレーム間での「ブロック単位の平均値」の平均値を保持
	m_means = means[0].clone() / num_embedframe;
	for (int i = 1; i < num_embedframe; i++) {
		m_means += means[i] / num_embedframe;
	}

	std::vector<cv::Mat> t_variance(num_embedframe);
	cv::pow((means[0] - m_means), 2, t_variance[0]);
	cv::Mat variance = t_variance[0].clone();
	for (int i = 1; i < num_embedframe; i++) {
		cv::pow((means[i] - m_means), 2, t_variance[i]);
		variance += t_variance[i];
	}
	variance /= num_embedframe;
	// 作成終了



	// 変数宣言＋初期化
	std::vector<cv::Mat> Is_stop;
	std::vector<cv::Mat> Is_move;
	std::vector<cv::Mat> result_lumi;
	std::vector<cv::Mat> lumi_map;
	std::vector<cv::Mat> comp;

	for (int i = 0; i < num_embedframe; i++) {
		Is_stop.push_back(cv::Mat::zeros(cv::Size(FRAME_width / block_width, FRAME_height / block_height), CV_16SC1));
		Is_move.push_back(cv::Mat::zeros(cv::Size(FRAME_width / block_width, FRAME_height / block_height), CV_8UC1));
		result_lumi.push_back(cv::Mat::zeros(cv::Size(FRAME_width / block_width, FRAME_height / block_height), CV_32F));
		lumi_map.push_back(cv::Mat::zeros(cv::Size(FRAME_width / block_width, FRAME_height / block_height), CV_32F));
		comp.push_back(cv::Mat::zeros(cv::Size(FRAME_width / block_width, FRAME_height / block_height), CV_8UC1));

		for (int y = 0; y < FRAME_height / block_size; y++) {
			for (int x = 0; x < FRAME_width / block_size; x++) {
				lumi_map[i].at<float>(y, x) = -1;
			}
		}
	}



	// 動きベクトルの処理
	for (int i = num_embedframe - 1; i > 0; i--) {       // 先頭フレームの動きベクトルは必要ない
		for (int y = 0; y < FRAME_height / block_size; y++) {
			for (int x = 0; x < FRAME_width / block_size; x++) {
				if (Is_there_mv(mv_all, cframe + i)) {  // これなくても十分	
					std::pair<int, int> back_point = get_back_pos(mv_all, cframe + i, y, x);    //フレーム番号とptsがごっちゃになっていないか確認する(現在のフレームを返せばいいと思われ)


					// debug 
					if (i == 3 && x == 2) {
						int debug_aa = 0;

						debug_aa = back_point.second;
						std::cout << debug_aa << std::endl;
					}

					// debugend

					if (back_point.first != y || back_point.second != x) {                  // 他の画素位置に移動しているなら
						if (Is_stop[i].at<short>(y, x) == -1) {     // 既に移動していた場合
							Is_stop[i].at<short>(y, x) = 2;
						}
						else {
							Is_stop[i].at<short>(y, x) = 1;             // 移動前の位置にチェック
						}

						Is_stop[i - 1].at<short>(back_point.first, back_point.second) = -1;    // 移動先にもチェック
						Is_move[i].at<unsigned char>(y, x) = 1;          
					}



					//// debug
					//int debug = 0;
					//if (i == 4 && x <=6) {
					//	std::cout << "i == 4" << std::endl;
					//	for (int j = 0; j < 4; j++) {
					//		std::cout << Is_stop[i].at<short>(0, j) << std::endl;
					//	}
					//}
					//else if (i == 3 && x <= 6) {
					//	std::cout << "i == 3" << std::endl;
					//	for (int j = 0; j < 4; j++) {
					//		std::cout << Is_stop[i].at<short>(0, j) << std::endl;
					//	}
					//}
					//else if (i == 1 && x <= 6) {
					//	std::cout << "i == 1" << std::endl;
					//	for (int j = 0; j < 4; j++) {
					//		std::cout << Is_stop[i - 1].at<short>(0, j) << std::endl;
					//	}
					//}
					//else if (i == 0 && x <= 6) {
					//	std::cout << "i == 0" << std::endl;
					//	for (int j = 0; j < 4; j++) {
					//		std::cout << Is_stop[i].at<short>(0, j) << std::endl;
					//	}
					//}


					////debug end
				}
				else {
					// 継続条件を満たさないようにしてforを2つ抜ける
					y = FRAME_height / block_size - 1;
					x = FRAME_width / block_size - 1;
				}
			}
		}
	}


	//// debug
	//std::cout << Is_stop[3].at<short>(0, 2) << std::endl;
	//std::cout << Is_stop[2].at<short>(0, 1) << std::endl;
	//std::cout << Is_stop[1].at<short>(0, 3) << std::endl;

	//std::cout << "aa " << std::endl;
	//for (int i = 0; i < 4; i++) {
	//	std::cout << Is_stop[i].at<short>(0, 1) << std::endl;
	//}

	//// debug end

	//↑途中で上書きされたとしてもその情報は，lumi_mapには載らない(でも，外側のループをnum_embedframeにすれば，)


	// 20フレームの最後から順に動きベクトルを調べて20個の輝度を確保する
	int num;          // 現在の画素に割り当てるべき透かしビットを格納
	// lumi_map求める

	for (int y = 0; y < FRAME_height / block_size; y++) {       
		for (int x = 0; x < FRAME_width / block_size; x++) { 
			lumi_map[num_embedframe - 1].at<float>(y, x) = means[num_embedframe - 1].at<float>(y *  block_size, x *  block_size);
			comp[num_embedframe - 1].at<unsigned char>(y, x) = 1;

			int jump_flg = 0;   // 移動した際に移動先のIsstopに-1をつける．そのため最初のifで通った直後に次のループのelseifに引っかからないようにするため
			int temp_y = y;
			int temp_x = x;
			for (int i = num_embedframe - 1; i > 0; i--) {
				if (Is_stop[i].at<short>(temp_y, temp_x) == 1) {  // 他の画素に移動しているなら
					std::pair<int, int> back_point;
					back_point = get_back_pos(mv_all, cframe + i, temp_y, temp_x);
				
					lumi_map[i - 1].at<float>(y, x) = means[i - 1].at<float>(back_point.first *  block_size, back_point.second *  block_size);
					comp[i - 1].at<unsigned char>(back_point.first, back_point.second) = 1;
					temp_y = back_point.first;
					temp_x = back_point.second;
					jump_flg = 1;
				}
				else if (Is_stop[i].at<short>(temp_y, temp_x) == -1 && jump_flg != 1) {       // 移動せず上書きされたならこれ以上lumi_mapには何も読み込まない
					break;
				}
				else {
					lumi_map[i - 1].at<float>(y, x) = means[i - 1].at<float>(temp_y *  block_size, temp_x *  block_size);
					comp[i - 1].at<unsigned char>(temp_y, temp_x) = 1;
					jump_flg = 0;
				}
			}
		}
	}

	////debug

	   	   
	// 結局この段階でlumi_mapはどうなっているのか，すべて埋まっているのか，穴があるのか．．．
	// lumi_mapには，lumi_mapの画素位置ごとに先頭フレームからそのブロックがどのように移動したかが書かれている
	// 穴が発生する(-1に初期化されている)のは動きが発生したときのみ

	std::vector<float> lumi(num_embedframe, 0);         //　lumi_mapの各ブロックの輝度値を取り出して，計算する際にfloatにする必要がある
	int sum_stop = 0;
	float ave_lumi = 0;
	float var_lumi = 0;
	int no_embed_count = 0;
	int no_embed_pos_count = 0;

	for (int y = 0; y < FRAME_height / block_size; y++) {
		for (int x = 0; x < FRAME_width / block_size; x++) {
			// 初期化
			ave_lumi = 0;
			var_lumi = 0;
			sum_stop = 0;
			int no_embed_flg = 0;
			for (int i = num_embedframe - 1; i >= 0 ; i--) {          
				lumi[i] = 0;
				if (lumi_map[i].at<float>(y, x) == -1) {    
					no_embed_flg = 1;
					no_embed_pos_count++;
				}
			}

			if (no_embed_flg == 1) {       // 上書きされた場合
				no_embed_count++;
				continue;
			}


			num = (embed[(x) % BG_width + ((y) % BG_height)*BG_width] == '0') ? 0 : 1;

			for (int i = num_embedframe - 1; i >= 0; i--) {
				lumi[i] = lumi_map[i].at<float>(y, x);
				ave_lumi += lumi[i];
			}
			ave_lumi /= num_embedframe;
			for (int i = 0; i < num_embedframe; i++) {
				var_lumi += pow((lumi[i] - ave_lumi), 2);
			}
			var_lumi / num_embedframe;   // 分散の定義式:(要素-平均)^2 /要素数


			// 透かしビットに応じて計算
			if (num == 0) {
				//for (int t_delta = delta; t_delta >= 1; t_delta--) {   // deltaよりもどの程度分散が大きいかどうかで操作する量を決めている
				//	if (var_lumi >= t_delta * t_delta) {
				//		operate_lumi_for_zero(lumi, ave_lumi, var_lumi, t_delta);
				//		break;
				//	}
				//}
			}
			else {    // 透かしビットが1の時
				operate_lumi_for_one(lumi, ave_lumi, var_lumi, delta);
			}


			// 埋め込んだ結果をresult_lumiに格納
			result_lumi[num_embedframe - 1].at<float>(y, x) = lumi[num_embedframe - 1];
			int temp_y = y;
			int temp_x = x;

			for (int i = num_embedframe - 1; i > 0; i--) {
				std::pair<int, int> back_point;
				back_point = get_back_pos(mv_all, cframe + i, temp_y, temp_x);
				temp_y = back_point.first;
				temp_x = back_point.second;
				result_lumi[i - 1].at<float>(temp_y, temp_x) = lumi[i - 1];    
			}
		}
	}


	//埋め込み後フレームを返す
	std::vector<float> temp_debug(num_embedframe);
	std::vector<float> temp_debug2(num_embedframe);

	for (int i = 0; i < num_embedframe; i++) {
		for (int y = 0; y < FRAME_height / block_height; y++) {
			for (int x = 0; x < FRAME_width / block_width; x++) {
				if (comp[i].at<unsigned char>(y, x) == 0) {
					result_lumi[i].at<float>(y, x) = means[i].at<float>(y * block_height, x *  block_width);
				}
			}
		}

		for (int y = 0; y < FRAME_height / block_height; y++) {
			for (int x = 0; x < FRAME_width / block_width; x++) {
				for (int m = 0; m < block_height; m++) {
					for (int n = 0; n < block_width; n++) {
						deviations[i].at<float>(y * block_height + m, x * block_width + n) += result_lumi[i].at<float>(y, x);
					}
				}
			}
		}
	}

	for (int i = 0; i < num_embedframe; i++) {
		dst_luminance.push_back(deviations[i]);
		dst_luminance[i].convertTo(dst_luminance[i], CV_8UC1);

		//	cv::imshow("0", dst_luminance[i]);
		//cv::waitKey(200);	
	}


	std::cout << "上書きされて埋め込まれない箇所は" << no_embed_pos_count << "個" << std::endl;
	std::cout << "割り当てられなかった個数は" << no_embed_count << "個" << std::endl;
}


bool is_less_than(float i) {
	return ((i < average_thisfile) == 1);
}

bool is_more_than(float i) {
	return ((i > average_thisfile) == 1);
}

void operate_lumi_for_one(std::vector<float> &lumi, float average, float variance, int delta) {
	size_t index_max, index_min; // 最大、最小の要素の添え字
	size_t num_low_ave = 0;  // 平均よりも低い個数
	size_t num_high_ave = 0; //平均よりも高い個数
	float now_variance;

	int p_cnt = 0;
	int m_cnt = 0;

	for (int i = 0; i < num_embedframe; i++) {
		if ((lumi[i] >= average && p_cnt != num_embedframe / 2) || m_cnt == num_embedframe / 2) {
			lumi[i] += delta;  //δ加えたフレームの数
			p_cnt++;
		}
		else {
			lumi[i] -= delta;  //-δ加えたフレームの数
			m_cnt++;
		}
	}
}

void operate_lumi_for_zero(std::vector<float> &lumi, float average, float variance, int delta) {  // 平均を維持しつつ、標準偏差を分散未満にする関数
	// average, varianceはlumiで与えられる輝度値の平均と分散であり、deltaは埋め込み強度


	size_t index_max, index_min; // 最大、最小の要素の添え字
	size_t num_low_ave = 0;  // 平均よりも低い個数
	size_t num_high_ave = 0; //平均よりも高い個数
	float now_variance;
	std::vector<double> temp_lumi(20);

	average_thisfile = average;


	// 計算用の配列に格納
	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		temp_lumi[i] = lumi[i];
	}


	for (int limit_time = 0; limit_time < 30; limit_time++) {
		// 平均から最も遠い要素のインデックスを求める
		std::vector<double>::iterator itr_max = std::max_element(temp_lumi.begin(), temp_lumi.end());
		std::vector<double>::iterator itr_min = std::min_element(temp_lumi.begin(), temp_lumi.end());
		index_max = std::distance(temp_lumi.begin(), itr_max);
		index_min = std::distance(temp_lumi.begin(), itr_min);

		num_low_ave = std::count_if(temp_lumi.begin(), temp_lumi.end(), is_less_than);
		num_high_ave = std::count_if(temp_lumi.begin(), temp_lumi.end(), is_more_than);
		now_variance = 0;

		temp_lumi[index_max]--;
		temp_lumi[index_min]++;

		for (int k = 0; k < (end(temp_lumi) - begin(temp_lumi)); k++) {
			now_variance += (temp_lumi[k] - average) * (temp_lumi[k] - average);
		}

		if ((now_variance <= (variance * (10 - delta) / 10)) || (now_variance <= (variance - delta * delta))) {
			break;
		}
	}


	// 元に戻す
	for (int i = 0; i < end(lumi) - begin(lumi); i++) {
		lumi[i] = temp_lumi[i];
	}
}



int ptob(int pixel_pos) {      
	return pixel_pos * block_size / motionvector_block_size;     // この辺も　もう一度チェックする
}

int btop(int block_pos) {   // 絶対値が０でないが１６未満の場合は正の整数なら１，負の整数ならー１を返す
	if (block_pos < 0 && block_pos > - 16) {
		return -1;
	}
	else if(block_pos > 0 && block_pos < 16) {
		return 1;
	}

	return block_pos  / block_size;
}


// mv_all[未来の座標] = 未来の座標 - 過去の座標
std::pair<int, int > get_back_pos(std::vector<mv_class>& mv_all, int frame, int y, int x) {
	// フレーム番号と座標を与えるとその次の座標を返す)
	// 何も動いていなければ元の座標を返す
	int bl_y = ptob(y);
	int bl_x = ptob(x);

	std::pair<int, int> back_point;

	if (motionvector_block_size == 16) {       // フレームの縦幅が1080の時はマクロブロックサイズ16では割り切れず，8画素分余る．その場合，そのまま座標を返す
		if (y == 134) {
			back_point.first = y;
			back_point.second = x;

			return back_point;
		}
	}

	//// debug
	//int de_bly = ptob(0);
	//int de_blx = ptob(2);
	//int debug_tempy = mv_all[3].y_vector.at<MV_DETA_TYPE>(de_bly, de_blx);
	//int debug_tempx = mv_all[3].x_vector.at<MV_DETA_TYPE>(de_bly, de_blx);

	//back_point = std::make_pair(0 - btop(debug_tempy), 2 - btop(debug_tempx));

	//std::cout << back_point.second << std::endl;

	//for (int i = 0; i < 5; i++) {
	//	mv_all[9].x_vector;
	//}


	////end debug

	int temp_y = mv_all[frame % num_embedframe].y_vector.at<MV_DETA_TYPE>(bl_y, bl_x);
	int temp_x = mv_all[frame % num_embedframe].x_vector.at<MV_DETA_TYPE>(bl_y, bl_x);


	back_point = std::make_pair(y - btop(temp_y), x - btop(temp_x));


	// 現状のエラーをとりあえず改善するため
	if (back_point.first < 0 || back_point.first >= (FRAME_height / block_size)) {       // ここが少しおかしいか
		back_point.first = y;
	}
	if (back_point.second < 0 || back_point.second >= (FRAME_width / block_size)) {     // ここが少しおかしいか
		back_point.second = x;
	}

	return back_point;
}


bool Is_there_mv(std::vector<mv_class> &mv_all, int frame) {  // ダミー場合はfalseを返す

	if (mv_all[frame % num_embedframe].frame_index == -1) {
		return false;
	}
	else {
		return true;
	}
}