package regression.online.learner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import regression.online.model.Prediction;
import regression.online.util.CholeskyDecom;
import regression.online.util.MatrixOp;
import regression.online.util.MatrixPrinter;


public class GPWindowedGaussianKernelAvgMean extends GPWindowedBase {
	
	double running_window_sum;
	
	public GPWindowedGaussianKernelAvgMean(int input_width, int window_size, double sigma_y, double sigma_w, boolean verbouse, int tuning_mode, boolean update_inhibator) {
		super(false, input_width, window_size, sigma_y, sigma_w, tuning_mode, true, update_inhibator);
		
		this.verbouse = verbouse;
		
		running_window_sum = 0;
	}
	
	@Override
	public Prediction predict(double[][] org_dp) throws Exception {
		
		double[][] dp = scale_input(org_dp);
		
		double[][] spare_column = new double[w_size][1];
		
		if(slide) {
			for(int ctr = 0; ctr < w_size; ctr++) {
				spare_column[ctr][0] = gaussian_kernel(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 1 - ctr][0] = gaussian_kernel(dp_window[w_end - 1 - ctr], dp);
			}
			for(; ctr < w_size - 1; ctr++) {
				spare_column[w_size - 1 - ctr][0] = 0;
			}
		}
		
		double[][] temp_column = MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv);
		
		double y = mean_func(dp);
		
		for(int ctr = 0; ctr < n; ctr++)
			y += temp_column[0][w_size - n + ctr]*(responses[(w_start + ctr) % w_size][0] - mean_func(dp));
		
		double kernel_measure = gaussian_kernel(dp, dp) - Math.pow(Math.E, 2*latent_log_hyperparams[0]);
		double predictive_variance = kernel_measure - MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv), spare_column)[0][0]; 
		double predictive_deviance;
		
		if(predictive_variance < 10E-20) predictive_deviance = 0;  // suspicious!
		else predictive_deviance = Math.sqrt(predictive_variance);
		
//		System.out.println(mean_func(dp) + " + " + (y - mean_func(dp)) + ", with predictive variance: " + predictive_variance);
		
		return new Prediction(target_postscaler(y), target_postscaler(predictive_deviance));
	}
	
	@Override
	public void update(double[][] org_dp, double org_y, Prediction prediction) throws Exception {
		
		if(update_count > burn_in_count && update_inhibator) return;
		
		double[][] dp = scale_input(org_dp);
		double y = target_prescaler(org_y);
		
		int index = getIndexForDp(dp);
		
		if(index != -1) {
			// reject the update
			// avg the response for the duplicate point
			double temp = responses[index][0];
			responses[index][0] = (y + responses[index][0])/2.0;
			
			running_window_sum -= temp;
			running_window_sum += responses[index][0];
			
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
		
		shrunk_inv = MatrixOp.mat_add(shrunk_inv, MatrixOp.scalarmult(MatrixOp.mult(spare_column, MatrixOp.transpose(spare_column)), -1.0/spare_var));
		
		if(slide) {
			for(int ctr = 1; ctr < w_size; ctr++) {
				spare_column[ctr-1][0] = gaussian_kernel(dp_window[(w_start + ctr) % w_size], dp);
			}
		}
		else {
			int ctr = 0;
			for(; ctr < n; ctr++) {
				spare_column[w_size - 2 - ctr][0] = gaussian_kernel(dp_window[w_end - 1 - ctr], dp);
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
		
		spare_var = gaussian_kernel(dp, dp);
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
		
//		k_inv = MatrixOp.fast_invert_psd(k);
		
		// sliding the dp_window
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			dp_window[w_end][ctr][0] = dp[ctr][0];
		}
		
		
		double temp = responses[w_end][0]; 
		responses[w_end][0] = y;
		//mean_responses[w_end] = mean_func(dp);
		
		if(slide) {
			running_window_sum -= temp;
			running_window_sum += y; 
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		}
		else {
			running_window_sum += y;
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
		
		update_running_means(org_dp, org_y);
		if(slide) update_prediction_errors(org_y, prediction.point_prediction);
		
		if(is_tuning_time()) { 
			
			revert_windows();
			revert_mean();
			
			update_scaling_factors();
			
			scale_windows();
			scale_mean();
			
			recompute_k();
			recompute_k_inv();
			
			double[][] y_u = new double[w_size][1];
			
			// creating response-mean vector
			
			for(int ctr = 0; ctr < w_size; ctr++)
				y_u[ctr][0] = responses[(w_start + ctr) % w_size][0] - running_window_sum/n;
			
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
				if(verbouse) System.out.println("gradient " + ctr + " " + gradient[ctr]);
		}
	}

//	private void fit_linear_mean_model(double[][] responses_vector) throws Exception {
//		
//		double design_matrix[][] = new double[feature_count][w_size];
//		
//		for(int ctr = 0; ctr < n; ctr++) {
//			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
//				design_matrix[ctr2][(w_start + ctr) % w_size] = dp_window[(w_start + ctr) % w_size][ctr2][0];
//			}
//		}
//		
//		coeff_u = MatrixOp.mult(MatrixOp.fast_invert_psd(MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix))), MatrixOp.mult(design_matrix, responses_vector));
//	}

//	private double mean_func(double[][] dp) throws Exception {
//		return MatrixOp.mult(MatrixOp.transpose(coeff_u), dp)[0][0];
//	}
	
	private void scale_mean() {
		running_window_sum = target_prescaler(running_window_sum);
	}

	private void revert_mean() {
		 running_window_sum = target_postscaler(running_window_sum);		
	}

	private double mean_func(double[][] dp) {
		return running_window_sum/n;
	}
}
