package regression.online.learner;

import java.util.Arrays;
import java.util.Random;

import regression.online.model.Prediction;
import regression.util.CholeskyDecom;
import regression.util.MatrixOp;
import regression.util.MatrixPrinter;


public class GPWindowedGaussianKernelOLSMean extends GPWindowedBase {
	
	double[] mean_responses; //mean_response;
	double[][] coeff_u; // coefficients of the mean function;

	
	public GPWindowedGaussianKernelOLSMean(int input_width, int window_size, double sigma_y, double sigma_w, boolean verbouse, int tuning_mode) {
		super(input_width, window_size, sigma_y, sigma_w, tuning_mode, true);
		
		this.verbouse = verbouse;
		
		mean_responses = new double[w_size];
		coeff_u = new double[3*input_width][1];
		
	}
	
	@Override
	public Prediction predict(double[][] dp) throws Exception {
		
		double[][] scaled_dp = scale_input(dp);
		
		double[][] spare_column = new double[w_size][1];
		
		if(slide) {
			for(int ctr = 0; ctr < w_size; ctr++) {
				spare_column[ctr][0] = gaussian_kernel(dp_window[(w_start + ctr) % w_size], scaled_dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 1 - ctr][0] = gaussian_kernel(dp_window[w_end - 1 - ctr], scaled_dp);
			}
			for(; ctr < w_size - 1; ctr++) {
				spare_column[w_size - 1 - ctr][0] = 0;
			}
		}
		
		double[][] temp_column = MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv);
		
		double y = mean_func(scaled_dp);
		
		for(int ctr = 0; ctr < n; ctr++) {
			y += temp_column[0][w_size - n + ctr]*(responses[(w_start + ctr) % w_size][0] - mean_responses[(w_start + ctr) % w_size]);
		}
		
		double kernel_measure = gaussian_kernel(scaled_dp, scaled_dp) - Math.pow(Math.E, 2*latent_log_hyperparams[0]);
		double predictive_variance = kernel_measure - MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv), spare_column)[0][0]; 
		double predictive_deviance;
		
		if(predictive_variance < 10E-20) predictive_deviance = 0;  // suspicious!
		else predictive_deviance = Math.sqrt(predictive_variance);
		
//		if(verbouse) System.out.println(mean_func(dp) + " + " + (y - mean_func(dp)) + ", with predictive variance: " + predictive_variance);
		
		return new Prediction(target_postscaler(y), target_postscaler(predictive_deviance));
	}
	
	@Override
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		double[][] scaled_dp = scale_input(dp);
		double scaled_y = target_prescaler(y);
		
		int index = getIndexForDp(scaled_dp);
		
		if(index != -1) {
			// reject the update
			// avg the response for the duplicate point
			
			responses[index][0] = (scaled_y + responses[index][0])/2.0;
			return;
		}
		
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
		
//		shrunk_inv = MatrixOp.mat_add(shrunk_inv, MatrixOp.scalarmult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), -1.0/spare_var));
		shrunk_inv = MatrixOp.mat_add(shrunk_inv, MatrixOp.scalardiv(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), -1*spare_var));
		
		if(slide) {
			for(int ctr = 1; ctr < w_size; ctr++) {
				spare_column[ctr-1][0] = gaussian_kernel(dp_window[(w_start + ctr) % w_size], scaled_dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 2 - ctr][0] = gaussian_kernel(dp_window[w_end - 1 - ctr], scaled_dp);
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
		
		spare_var = gaussian_kernel(scaled_dp, scaled_dp);
		k[w_size-1][w_size-1] = spare_var;
		
		// computing new k_inv
		
		double temp = MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), shrunk_inv), spare_column)[0][0];
		double spare_var_inv = (spare_var - temp);
		spare_var = 1/spare_var_inv;
				
		double[][] temp_upper_left = MatrixOp.mult(shrunk_inv, MatrixOp.identitiy_add(MatrixOp.scalardiv(MatrixOp.mult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), MatrixOp.transpose(shrunk_inv)), spare_var_inv), 1));
				
		spare_column = MatrixOp.scalardiv(MatrixOp.mult(shrunk_inv, spare_column), -1*spare_var_inv);		
		
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
		
//		double[][] k_inv_ref = null;
//		
//		try {
//			k_inv_ref = MatrixOp.fast_invert_psd(k);
//		}
//		catch (Exception e) {
//			System.out.println("SHIT");
//		}
//		
//		double deviance = MatrixOp.calc_deviance(k_inv_ref, k_inv);
//		System.out.println(deviance);
		
		
		// sliding the dp_window
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			dp_window[w_end][ctr][0] = scaled_dp[ctr][0];
		}
		
		responses[w_end][0] = scaled_y;
		mean_responses[w_end] = mean_func(scaled_dp);
		
		update_running_se(Math.abs(y - prediction.point_prediction));
		
		if(slide) {
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		}
		else {
			w_end = (w_end + 1) % w_size;
			
			if(w_start == w_end) slide = true;
		}
		
		count_dps_in_window();
		
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
		
		update_running_means(dp, y);
//		if(slide) update_prediction_errors(y, prediction.point_prediction);
		
		if(is_tuning_time()) {
			
			revert_windows();
//			revert_means();
			
			update_scaling_factors();
			
			scale_windows();
//			scale_means();
			
			recompute_k();
			recompute_k_inv();
			
			double[][] y_u = new double[w_size][1];
			
			// creating response-mean vector
			
			for(int ctr = 0; ctr < w_size; ctr++)
				y_u[ctr][0] = responses[(w_start + ctr) % w_size][0];
			
			fit_linear_mean_model(y_u);
			
			recompute_means();
			
			for(int ctr = 0; ctr < w_size; ctr++)
				y_u[ctr][0] -= mean_func(dp_window[(w_start + ctr) % w_size]);
			
			double marginal_lhood = get_likhood(y_u);
			
			if(verbouse) System.out.println("Pre-Optimization Hyperparams :");
			
			for(int ctr = 0; verbouse && ctr < latent_log_hyperparams.length; ctr++)
				System.out.println("parameter " + ctr + " " + Math.pow(Math.E, latent_log_hyperparams[ctr]));
			
			if(verbouse) System.out.println("Pre-Optimization Likelihood : " + marginal_lhood);
			
			double[] gradient = new double[latent_log_hyperparams.length];
			
			set_gradients(gradient, y_u);
			
			if(verbouse) System.out.println("Pre-Optimization Hyperparameter-Log Gradients :");
			for(int ctr = 0; verbouse && ctr < latent_log_hyperparams.length; ctr++)
				System.out.println("gradient " + ctr + " " + gradient[ctr]);
			
			if(verbouse) System.out.println("--------------------------");
			
			//if(true) return;
			
			optimize_hyperparams(y_u, marginal_lhood);
			//update_hyperparams_steepestasc(responses_minus_mean_vector, marginal_lhood);
			//update_hyperparams_rprop(responses_minus_mean_vector, marginal_lhood);
			
			if(verbouse) System.out.println("--------------------------");
			
			marginal_lhood = get_likhood(y_u);
			
			if(verbouse) System.out.println("Post-Optimization Hyperparams :");
			
			for(int ctr = 0; verbouse && ctr < latent_log_hyperparams.length; ctr++)
				System.out.println("parameter " + ctr + " " + Math.pow(Math.E, latent_log_hyperparams[ctr]));
			
			if(verbouse) System.out.println("Post-Optimization Likelihood : " + marginal_lhood);
			
			set_gradients(gradient, y_u);
			
			if(verbouse) System.out.println("Post-Optimization Hyperparameter-Log Gradients :");
			for(int ctr = 0; verbouse && ctr < latent_log_hyperparams.length; ctr++)
				System.out.println("gradient " + ctr + " " + gradient[ctr]);
		}
	}

	private void recompute_means() throws Exception {
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			mean_responses[ptr] = mean_func(dp_window[ptr]);
		}
		
	}

	private void scale_means() {
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			mean_responses[ptr] = target_prescaler(mean_responses[ptr]);
		}
		
	}

	private void revert_means() {

		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++) {
			mean_responses[ptr] = target_postscaler(mean_responses[ptr]);
		}
		
	}

	private void fit_linear_mean_model_basic(double[][] responses_vector) throws Exception {
		
		double design_matrix[][] = new double[feature_count][w_size];
		
		for(int ctr = 0; ctr < n; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				double entry = dp_window[(w_start + ctr) % w_size][ctr2][0];
				design_matrix[ctr2][(w_start + ctr) % w_size] = entry;
			}
		}
		
		double[][] x_x_t = MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix));
		
		coeff_u = MatrixOp.mult(MatrixOp.fast_invert_psd(x_x_t), MatrixOp.mult(design_matrix, responses_vector));
		
		boolean illegal_coeffs = false;
		for (int ctr = 0; ctr < coeff_u.length; ctr++) {
			if(Double.isNaN(coeff_u[ctr][0]) || Double.isInfinite(coeff_u[ctr][0])) {
				illegal_coeffs = true;
				break;
			}
		}
		
		if(illegal_coeffs) {
			x_x_t = MatrixOp.identitiy_add(x_x_t, 0.1);
			coeff_u = MatrixOp.mult(MatrixOp.fast_invert_psd(x_x_t), MatrixOp.mult(design_matrix, responses_vector));
		}
	}
	
	private void fit_linear_mean_model(double[][] responses_vector) throws Exception {
		
		double design_matrix[][] = new double[3*feature_count][w_size];
		
		for(int ctr = 0; ctr < n; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				double entry = dp_window[(w_start + ctr) % w_size][ctr2][0];
				design_matrix[ctr2*3][(w_start + ctr) % w_size] = Math.sqrt(entry);
				design_matrix[ctr2*3+1][(w_start + ctr) % w_size] = entry;
				design_matrix[ctr2*3+2][(w_start + ctr) % w_size] = entry*entry;
			}
		}
		
		double[][] x_x_t = MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix));
		
		coeff_u = MatrixOp.mult(MatrixOp.fast_invert_psd(x_x_t), MatrixOp.mult(design_matrix, responses_vector));
		
		boolean illegal_coeffs = false;
		for (int ctr = 0; ctr < coeff_u.length; ctr++) {
			if(Double.isNaN(coeff_u[ctr][0]) || Double.isInfinite(coeff_u[ctr][0])) {
				illegal_coeffs = true;
				break;
			}
		}
		
		if(illegal_coeffs) {
			x_x_t = MatrixOp.identitiy_add(x_x_t, 0.1);
			coeff_u = MatrixOp.mult(MatrixOp.fast_invert_psd(x_x_t), MatrixOp.mult(design_matrix, responses_vector));
		}
	}

	private double mean_func_basic(double[][] dp) throws Exception {
		
		double result = 0;
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			double entry = dp[ctr][0];
			result += coeff_u[ctr][0]*entry;
		}
		
		return result;
	}
	
	private double mean_func(double[][] dp) throws Exception {
		
		double result = 0;
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			double entry = dp[ctr][0];
			result += coeff_u[ctr*3][0]*Math.sqrt(entry);
			result += coeff_u[ctr*3+1][0]*entry;
			result += coeff_u[ctr*3+2][0]*entry*entry;
		}
		
		return result;
	}
	
	@Override
	public void update_scaling_factors() {
		
		if(target_mean != 0)
			target_scaler = 1/(target_mean);
		
		for (int ctr = 0; ctr < input_means.length; ctr++) {
			if(input_means[ctr] == 0) continue;
			input_scaler[ctr] = 1/(input_means[ctr]);
		}
		
	}

}
