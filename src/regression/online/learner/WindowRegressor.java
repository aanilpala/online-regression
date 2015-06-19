package regression.online.learner;

import regression.online.util.MatrixOp;

public abstract class WindowRegressor extends Regressor {

	public int w_size; // sliding window size
	public int w_start, w_end;
	public int n; // number of data points in the window
	double[][][] dp_window;
	double[][] responses;
	boolean slide;
	int update_count;
	
	public void count_dps_in_window() {
		if(slide) n = w_size;
		else n = w_end - w_start;
	}
	
	public WindowRegressor(boolean map2fs, int input_width, int window_size) {
		
		super(map2fs, input_width);
		
		w_size = window_size;
		dp_window = new double[w_size][feature_count][1];
		responses = new double[w_size][1];
		
		slide = false;
		w_start = 0;
		w_end = 0;
		update_count = 0;
	}
	
	protected int getIndexForDp(double[][] dp) throws Exception {
		
		for(int ctr = 0; ctr < n; ctr++)
			if(MatrixOp.isEqual(dp, dp_window[(w_start + ctr) % w_size])) return (w_start + ctr) % w_size;
		
		return -1;
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
	public int get_burn_in_number() {
		return w_size;
	}
}
