package regression.online.learner;

import java.io.IOException;
import java.util.Arrays;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;
import regression.online.util.MatrixPrinter;

public class GPWindowedHPTuningSGD extends WindowRegressor {

	double[][] k; // gram matrix
	double[][] k_inv; // inverse gram matrix	
	
	// hyperparams
	double a; //measurement_precision;
	double b; //weight_precision;
	double[] l; // length-scale array;
	
	double init_length_scale = 10000;
	//double hyperparam_learning_rate = 0.000001;
	double hyper_param_update_freq = w_size;
	
	public GPWindowedHPTuningSGD(int input_width, double signal_stddev, double weight_stddev) {
		super(false, input_width);
		
		name = "GPWindowed";
		
		k = new double[w_size][w_size]; 
		k_inv = new double[w_size][w_size];
		
		l = new double[input_width];
		
		for(int ctr = 0; ctr < input_width; ctr++)
			l[ctr] = init_length_scale;
		
		a = 1/(signal_stddev*signal_stddev);
		b = 1/(weight_stddev*weight_stddev);
		
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) {
					k[ctr][ctr2] = (1/b + 1/a);
					k_inv[ctr][ctr2] = 1/(1/b + 1/a);
				}
				else {
					k[ctr][ctr2] = 0;
					k_inv[ctr][ctr2] = 0;
				}
			}
		}
		
	}
	
	@Override
	public Prediction predict(double[][] dp) throws Exception {
		
		double[][] spare_column = new double[w_size][1];
		
		count_dps_in_window();
		
		if(slide) {
			for(int ctr = 0; ctr < w_size; ctr++) {
				spare_column[ctr][0] = kernel_func(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 1 - ctr][0] = kernel_func(dp_window[w_end - 1 - ctr], dp);
			}
			for(; ctr < w_size - 1; ctr++) {
				spare_column[w_size - 1 - ctr][0] = 0;
			}
		}
		
		double[][] temp_column = MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv);
		
		double y = 0;
		
		for(int ctr = 0; ctr < n; ctr++)
			y += temp_column[0][w_size - n + ctr]*responses[(w_start + ctr) % w_size][0];
		
		double predictive_deviance = kernel_func(dp, dp) -  MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv), spare_column)[0][0];
		
		return new Prediction(y, y + 1.96*predictive_deviance, y - 1.96*predictive_deviance);
		
		
	}
	
	@Override
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		double[][] shrunk = new double[w_size-1][w_size-1];
		double[][] shrunk_inv = new double[w_size-1][w_size-1];
		double[][] spare_column = new double[w_size-1][1];
		double spare_var;
		
		// shrinking
		// obtaining k_minus_hat
		for(int ctr = 1; ctr < w_size; ctr++)
			shrunk[ctr-1] = Arrays.copyOfRange(k[ctr], 1, w_size);
		
		for(int ctr = 1; ctr < w_size; ctr++) 
			shrunk_inv[ctr-1] = Arrays.copyOfRange(k_inv[ctr], 1, w_size);
		
		for(int ctr = 1; ctr < w_size; ctr++)
			spare_column[ctr-1][0] = k_inv[ctr][0];
		
		spare_var = k_inv[0][0];
		
		shrunk_inv = MatrixOp.mat_add(shrunk_inv, MatrixOp.scalarmult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), -1.0/spare_var));
		
		count_dps_in_window();
		
		if(slide) {
			for(int ctr = 1; ctr < w_size; ctr++) {
				spare_column[ctr-1][0] = kernel_func(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 2 - ctr][0] = kernel_func(dp_window[w_end - 1 - ctr], dp);
			}
			for(; ctr < w_size - 1; ctr++) {
				spare_column[w_size - 2 - ctr][0] = 0; //kernel_func(null, dp);;
			}
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
		
		spare_var = kernel_func(dp, dp);
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
		
		responses[w_end][0] = y;
		
		if(slide) {
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		}
		else {
			w_end = (w_end + 1) % w_size;
			
			if(w_start == w_end) slide = true;
		}
		
//		System.out.println("post-update");
//		MatrixPrinter.print_matrix(k);
//		System.out.println("post-update_inv");
//		MatrixPrinter.print_matrix(k_inv);		
//		
//		System.out.println("--------------------------------");
		
//		try {
//			System.in.read();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		update_count++;
		
		if(slide && ((update_count - w_size) % hyper_param_update_freq == 0)) 
			update_hyperparams();
	}

	private void update_hyperparams() throws Exception {
		
		// we have a, b and length-scales as hyperparams to tune
		
		System.out.println("pre-tuning b: " + b);
		
		count_dps_in_window();
		
		double[][] derivative_cov_matrix = new double[w_size][w_size];
		double[][] y = new double[w_size][1];
		double gradient;
		
		// creating y vector
		
		if(slide) {
			for(int ctr = 0; ctr < w_size; ctr++)
				y[ctr][0] = responses[(w_start + ctr) % w_size][0];
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				y[w_size - 1 - ctr][0] = responses[w_end - 1 - ctr][0];
			}
			for(; ctr < w_size - 1; ctr++) {
				y[w_size - 1 - ctr][0] = 0;
			}
		}
		
		// gradient ascent
		
		double sigma_w = Math.sqrt(1/b);
		double delta = Double.MAX_VALUE;
		double min_delta = 0.01;
		double step_size = 0.1;
		double accelerator = 1.25;
		double decay = 0.8;
		int sign = 0;
		
		System.out.println("tuning...");
		
		//while(learning_rate > 0 && delta > min_delta) {
		while(true) {
			double old_sigma_w = sigma_w;
			
			for(int ctr = 0; ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
					if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2*sigma_w; 
					else derivative_cov_matrix[ctr][ctr2] = k[ctr][ctr2]*2/sigma_w;
				}
			}
			
//			double gradient_1 = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix));
//			double gradient_2 = 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
//			double gradient_3 = 0;//-1/sigma_w; 
//			double gradient_test = gradient_1 + gradient_2 + gradient_3;
			
			double[][] alpha = MatrixOp.mult(k_inv, y);
			gradient = 0.5*MatrixOp.trace(MatrixOp.mult(MatrixOp.mat_subtract(MatrixOp.mult(alpha, MatrixOp.transpose(alpha)), k_inv), derivative_cov_matrix)); // - 10/sigma_w;
			
			
			
			sigma_w += step_size*gradient;
			b = 1/(sigma_w*sigma_w);
			
			if(sign == 0) sign = (int) Math.signum(gradient);
			else {
				sign *= (int) Math.signum(gradient);
				if(sign >= 1) {
					step_size *= accelerator;
					sign = 0;
				}
				else {
					step_size *= decay;
					sign = 0;
				}
			}
			
			delta = Math.abs(sigma_w - old_sigma_w);
			
			//System.out.println(gradient + " / " + sigma_w);
			//System.out.println(gradient_test);
			
			//recomputing covariance matrix and its inverse
			//assuming sliding is already enabled
			for(int ctr = 0; slide && ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
					k[ctr][ctr2] = kernel_func(dp_window[(w_start+ctr) % w_size], dp_window[(w_start+ctr2) % w_size]);
				}
			}
			
			k_inv = MatrixOp.fast_invert_psd(k);
			
			if(Math.abs(gradient) < 0.001) {
				System.out.println(gradient);
				break; 
			}
		}
		
		System.out.println("post-tuning b: " + b);
		System.out.println("-------------------------");
		
//		try {
//			System.in.read();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

	}

	private double kernel_func(double[][] dp1, double dp2[][]) {
		
		double sum, kernel_measure;
		
		if(dp1 == null) {
			sum = 0;
			
			for(int ctr = 0; ctr < dp2.length; ctr++) {
				double dif = dp2[ctr][0];
				sum += dif*dif/l[ctr];
			}
			
			kernel_measure = Math.pow(Math.E, -0.5*sum)/b;
		}
		else {
			int dim = dp1.length;
			sum = 0;
			
			for(int ctr = 0; ctr < dim; ctr++) {
				double dif = dp1[ctr][0] - dp2[ctr][0];
				sum += dif*dif/l[ctr];
			}
			
			if(sum == 0) kernel_measure = 1/b + 1/a;
			else kernel_measure = Math.pow(Math.E, -0.5*sum)/b;
		}
		
		return kernel_measure;
	}
}
