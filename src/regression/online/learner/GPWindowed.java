package regression.online.learner;

import java.util.Arrays;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;
import regression.online.util.MatrixPrinter;

public class GPWindowed extends WindowRegressor {

	double[][] k; // gram matrix
	double[][] k_inv; // inverse gram matrix
//	double[][] temp_matrix;
//	double[][] temp_matrix_inv;
//	double[][] temp_col;
	//double[][] f_t;
	
	double a; //measurement_precision;
	double b; //weight_precision;
	
	public GPWindowed(int input_width, double signal_stddev, double weight_stddev) {
		super(false, input_width);
		
		k = new double[w_size][w_size]; 
		k_inv = new double[w_size][w_size]; 
//		temp_matrix = new double[w_size-1][w_size-1];
//		temp_matrix_inv = new double[w_size-1][w_size-1];
//		temp_col = new double[w_size-1][1];
		//f_t = new double[window_size-1][1];
		
		feature_count = input_width;
		
		a = 1/(signal_stddev*signal_stddev);
		b = 1/(weight_stddev*weight_stddev);
		
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) {
					k[ctr][ctr2] = (1 + 1/a);
					k_inv[ctr][ctr2] = 1/(1.0 + 1/a);
				}
				else {
					k[ctr][ctr2] = 0;
					k_inv[ctr][ctr2] = 0;
				}
			}
		}
		
	}
	
	@Override
	public Prediction predict(double[][] dp) {
		return new Prediction();
	}
	
	@Override
	public void update(double[][] dp, double y, Prediction prediction) {
		
		double[][] shrunk = new double[w_size-1][w_size-1];
		double[][] shrunk_inv = new double[w_size-1][w_size-1];
		double[][] spare_column = new double[w_size-1][1];
		double spare_var;
		
		System.out.println("pre-update");
		MatrixPrinter.print_matrix(k);
		System.out.println("pre-update_inv");
		MatrixPrinter.print_matrix(k_inv);
		
		// shrinking
		// obtaining k_minus_hat
		for(int ctr = 1; ctr < w_size; ctr++) {
			shrunk[ctr-1] = Arrays.copyOfRange(k[ctr], 1, w_size);
		}
		
		
		// computing k_minus_hat_inv
		for(int ctr = 1; ctr < w_size; ctr++) {
			shrunk_inv[ctr-1] = Arrays.copyOfRange(k_inv[ctr], 1, w_size);
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			spare_column[ctr-1][0] = k_inv[ctr][0];
		}
		
		spare_var = k_inv[0][0];
		
		shrunk_inv = MatrixOp.mat_add(shrunk_inv, MatrixOp.scalarmult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), -1.0/spare_var));
		
		System.out.println("shrunk");
		MatrixPrinter.print_matrix(shrunk);
		System.out.println("shrunk-inv");
		MatrixPrinter.print_matrix(shrunk_inv);
		
		// expanding
		// obtaining (new) k
		
		count_dps_in_window();
		
		if(slide) {
			for(int ctr = 1; ctr < w_size; ctr++) {
				spare_column[ctr-1][0] = kernel_func(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ptr = w_end;
			for(int ctr = 0; ctr < n; ptr--,  ctr++) {
				System.out.println(w_size - w_end + ctr - 1 + "-" + (w_end - ctr - 1));
				spare_column[w_size - w_end + ctr - 1][0] = kernel_func(dp_window[w_end - ctr - 1], dp);
			}
//			for(; ptr > 0; ptr--) {
//				spare_column[w_size - ptr][0] = 0;
//			}
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			for(int ctr2 = 1; ctr2 < w_size; ctr2++) {
				k[ctr-1][ctr2-1] = shrunk[ctr-1][ctr2-1];
			}
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			k[ctr-1][w_size-1] = spare_column[ctr-1][0];
			k[w_size-1][ctr-1] = spare_column[ctr-1][0];
		}
		
		spare_var = kernel_func(dp, dp) + 1/a;
		k[w_size-1][w_size-1] = spare_var;
		
		// computing new k_inv
		
		spare_var = 1.0/(spare_var - MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), shrunk_inv), spare_column)[0][0]);
		
		double[][] temp_upper_left = MatrixOp.mult(shrunk_inv, MatrixOp.identitiy_add(MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), MatrixOp.transpose(shrunk_inv)), spare_var), 1));
		
		spare_column = MatrixOp.scalarmult(MatrixOp.mult(shrunk_inv, spare_column), -1*spare_var);
		
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			for(int ctr2 = 1; ctr2 < w_size; ctr2++) {
				k_inv[ctr-1][ctr2-1] = temp_upper_left[ctr-1][ctr2-1];
			}
		}
		
		for(int ctr = 1; ctr < w_size; ctr++) {
			k_inv[ctr-1][w_size-1] = spare_column[ctr-1][0];
			k_inv[w_size-1][ctr-1] = spare_column[ctr-1][0];
		}
		
		k_inv[w_size-1][w_size-1] = spare_var;

		// sliding the dp_window
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			dp_window[w_end][ctr][0] = dp[ctr][0];
		}
		
		if(slide) {
			responses[w_end] = y;
			
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		}
		else {
			w_end = (w_end + 1) % w_size;
			
			if(w_start == w_end) slide = true;
		}
		
		
		System.out.println("post-update");
		MatrixPrinter.print_matrix(k);
		System.out.println("post-update_inv");
		MatrixPrinter.print_matrix(k_inv);		
		
		System.out.println("--------------------------------");
	}

	private static double kernel_func(double[][] dp1, double dp2[][]) {
		
		int dim = dp1.length;
		long sum = 0;
		
		for(int ctr = 0; ctr < dim; ctr++) {
			double dif = (dp1[ctr][0] - dp2[ctr][0]);
			sum += dif*dif*1000000*1000000;
		}
		
		double aa = Math.sqrt(sum/(1000000.0*1000000.0));
		
		double kernel_measure = Math.pow(Math.E, -0.5*(Math.sqrt(sum/(1000000.0*1000000.0))));
		
		return kernel_measure;
	}
	
	
	

}
