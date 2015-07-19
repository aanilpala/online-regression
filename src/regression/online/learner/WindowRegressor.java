package regression.online.learner;

import regression.util.MatrixOp;

public abstract class WindowRegressor extends OnlineRegressor {

	public int w_size; // sliding window size
	public int w_start, w_end;
	public int n; // number of data points in the window
	double[][][] dp_window;
	double[][] responses;
	double[] residual_window;
//	int successive_error_increase_count = 0;
	double running_se;
	boolean high_error_flag;
	int tuning_countdown;
	
	boolean slide;
	
	int tuning_mode;
	int tuning_freq;
	
	public void count_dps_in_window() {
		if(slide) n = w_size;
		else n = w_end - w_start;
	}
	
	public WindowRegressor(boolean map2fs, int input_width, int window_size, int tuning_mode) {
		
		super(map2fs, input_width);
		
		this.name += "WS" + window_size;
		
		this.tuning_mode = tuning_mode;
		
		w_size = window_size;
		tuning_freq = 2*w_size;
		burn_in_count = window_size;
		dp_window = new double[w_size][feature_count][1];
		responses = new double[w_size][1];
		
		residual_window = new double[w_size];
		
		slide = false;
		w_start = 0;
		w_end = 0;
		update_count = 0;
		
		running_se = 0;
		
		high_error_flag = false;
		tuning_countdown = w_size;
	}
	
	protected int getIndexForDp(double[][] dp) throws Exception {
		
		for(int ctr = 0; ctr < n; ctr++)
			if(MatrixOp.isEqual(dp, dp_window[(w_start + ctr) % w_size])) return (w_start + ctr) % w_size;
		
		return -1;
	}
	
	protected void scale_windows() {
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			dp_window[ptr] = scale_input(dp_window[ptr]);
			responses[ptr][0] = target_prescaler(responses[ptr][0]);
		}
		
	}
	
	protected void revert_windows() {
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			dp_window[ptr] = revert_input(dp_window[ptr]);
			responses[ptr][0] = target_postscaler(responses[ptr][0]);
		}
		
	}
	
	protected double[][] get_weights_cov_matrix() throws Exception {
		
		if(!slide) throw new Exception("Slide should be enabled for this op!");
		
		double[][] design_matrix = new double[w_size][feature_count];
		double[][] weight_cov = new double[w_size][feature_count];
		double[][] ones = new double[w_size][1];
		
		for(int ptr = w_start, ctr = 0; ctr < w_size; ptr = (ptr + 1) % w_size, ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				design_matrix[ptr][ctr2] = dp_window[ptr][ctr2][0];
			}
		}
		
		for(int ctr = 0; ctr < w_size; ctr++)
			ones[ctr][0] = 1;
		
		design_matrix = MatrixOp.mat_subtract(design_matrix, MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.mult(ones, MatrixOp.transpose(ones)), design_matrix), 1.0/w_size));
		weight_cov = MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.transpose(design_matrix), design_matrix), 1.0/w_size);
		
		return weight_cov;
		
	}
	
	@Override
	public boolean is_tuning_time() {
		if(!slide) return false;
		else if(tuning_mode == -2) return false;
		else if(tuning_mode == -1) return (update_count == w_size);
		else if(tuning_mode == 0) return (update_count - w_size) % tuning_freq == 0;
		else if(tuning_mode == 1) {
			if(update_count == w_size) return true;
			
			if(high_error_flag) {
				tuning_countdown--;
			}
			
			if(tuning_countdown == 0) {
				tuning_countdown = -1;
				high_error_flag = false;
				return true;
			}
			else return false;
			
		}
		else return false;
	}
	
	@Override
	public boolean was_just_tuned() {
		if(!slide) return false;
		else if(tuning_mode == -2) return false;
		else if(tuning_mode == -1) return (update_count == w_size);
		else if(tuning_mode == 0) return (update_count - w_size) % tuning_freq == 0;
		else if(tuning_mode == 1) return (update_count == w_size) || tuning_countdown == -1;
		else return false;
	}
	
//	@Override
//	public boolean high_error() {
//		if(successive_error_increase_count > w_size/2) {
//			successive_error_increase_count = 0;
//			prediction_count_since_last_update = 0;
//			return true;
//		}
//		else return false;
//		
//	}
	
	
	public void update_running_se(double residual) {
		
		if(tuning_countdown == -1) tuning_countdown = w_size;
		
		double old_running_se = running_se;
		double dropped = 0.0;
		
		if(!slide) {
			residual_window[w_end] = residual;
			running_se += residual*residual;
		}
		else {
			dropped = residual_window[w_end];
			residual_window[w_end] = residual;
			running_se += (residual*residual - dropped*dropped);
		}
		
//		if(update_count > 995 && update_count < 1005) {
//			System.out.println((n*residual*residual)/old_running_se);
//		}
		
		
//		double error_increase_ratio = (n*residual*residual)/
		
//		System.out.println(old_running_se + "-" + residual*residual + "-" + dropped*dropped);
		if(running_se < 0) running_se = 0.0;
		
		if(slide && n*residual*residual > 1 && 50*old_running_se < n*residual*residual) {
			high_error_flag = true;
//			System.out.println(old_running_se + " - " + n*residual*residual);
		}
	}

}
