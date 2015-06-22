package regression.online.learner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import regression.online.util.MatrixOp;

public abstract class GPWindowedBase extends WindowRegressor{

	double[][] k; // gram matrix
	double[][] k_inv; // inverse gram matrix	
	
	// hyperparams
	double[] latent_log_hyperparams; // length-scale array;
	int hyperparams_count;
	
	double init_lengthscale = 1;
		
	double hyper_param_update_freq = 100*w_size;
		
	Random rand = new Random();
	
	public GPWindowedBase(boolean map2fs, int input_width, int window_size, double sigma_y, double sigma_w) {
		super(map2fs, input_width, window_size);
		
		k = new double[w_size][w_size]; 
		k_inv = new double[w_size][w_size];
		
		hyperparams_count = feature_count + 2;
		latent_log_hyperparams = new double[hyperparams_count];
		
		latent_log_hyperparams[0] = Math.log(sigma_y);
		latent_log_hyperparams[1] = Math.log(sigma_w);
		
		for(int ctr = 2; ctr < hyperparams_count; ctr++) {
			latent_log_hyperparams[ctr] = Math.log(init_lengthscale);
		}		
		
		double val = Math.pow(Math.E, 2*latent_log_hyperparams[0]) + Math.pow(Math.E, 2*latent_log_hyperparams[1]);
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) {
					k[ctr][ctr2] = val;
					k_inv[ctr][ctr2] = 1/val;
				}
				else {
					k[ctr][ctr2] = 0;
					k_inv[ctr][ctr2] = 0;
				}
			}
		}
	}
	
	protected void set_gradients(double[] gradient, double[][] y) throws Exception {
		
		double[][] derivative_cov_matrix = new double[w_size][w_size];
		
		// gradient log sigma_y
				for(int ctr = 0; ctr < w_size; ctr++) {
					for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
						if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2.0*Math.pow(Math.E, 2*latent_log_hyperparams[0]); 
						else derivative_cov_matrix[ctr][ctr2] = 0;
					}
				}
		
		gradient[0] = 0; //-0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
				
		// gradient log sigma_w
		for(int ctr = 0; ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2.0*Math.pow(Math.E, 2*latent_log_hyperparams[1]); //2.0*hyperparams[1]; 
				else derivative_cov_matrix[ctr][ctr2] = (2.0)*k[ctr][ctr2]; // (2.0/hyperparams[1])*k[ctr][ctr2];
			}
		}
					
		gradient[1] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
					
		// lengthscale gradients
		for(int ctr3 = 2; ctr3 < hyperparams_count; ctr3++) {
			for(int ctr = 0; ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
					if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 0;
					else {
						double dif = dp_window[(w_start + ctr) % w_size][ctr3-2][0] - dp_window[(w_start + ctr2) % w_size][ctr3-2][0];
						derivative_cov_matrix[ctr][ctr2] = dif*dif*Math.pow(Math.E, -2*latent_log_hyperparams[ctr3])*k[ctr][ctr2]; //(((dif*dif))/(Math.pow(hyperparams[ctr3], 3)))*k[ctr][ctr2];
					}
				}
			}
			gradient[ctr3] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
		}
		
		
	}
	protected void optimize_hyperparams(double[][] y_u, double init_likhood) throws Exception {
		
		// optimizer parameters
		double default_step_size = 1;
		double min_step_size = 10E-10;
		double decayer = 0.1;
		double step_size;
		double max_approx_optima_gradient = 0.00000001;
		boolean progression = true;
		int reset_counter = 0;
		int reset_limit = 5;
		
		double[] gradient = new double[hyperparams_count];
		double[] prev_sane_hyperparams = new double[hyperparams_count];
		double[] target_hyperparams = new double[hyperparams_count];
		
		double cur_likhood = init_likhood;
		double target_likhood = cur_likhood;
		target_hyperparams = Arrays.copyOf(latent_log_hyperparams, hyperparams_count);
		
		while(true) {
			
			if(!progression) {
				if(reset_counter >= reset_limit) {
					cur_likhood = target_likhood;
					latent_log_hyperparams = Arrays.copyOf(target_hyperparams, hyperparams_count);
					break;
				}
				else {
					System.out.println("Resetting...");
					while(true) {
						// pick random values
						for(int ctr = 2; ctr < hyperparams_count; ctr++)
							latent_log_hyperparams[ctr] = rand.nextDouble();
							
						try {
							recompute_k();
							recompute_k_inv();
						}
						catch (Exception e) {
							continue;
						}
						
						double new_likhood = get_likhood(y_u);
						if(Double.isNaN(new_likhood) || Double.isInfinite(new_likhood)) continue;
						else {
							cur_likhood = new_likhood;
							reset_counter++;
							break;
						}
						
					}
				}
			}
			
			set_gradients(gradient, y_u);
			
			boolean optima_found = true;
			for (int ctr = 0; ctr < gradient.length; ctr++) {
				if(Math.abs(gradient[ctr]) > max_approx_optima_gradient) {
					optima_found = false;
					break;
				}
			}
			
			if(optima_found && cur_likhood >= target_likhood) break;
			
			step_size = default_step_size;
			progression = false;
			
			while(true) {
				
				if(step_size < min_step_size) break;
				
				prev_sane_hyperparams = Arrays.copyOf(latent_log_hyperparams, hyperparams_count); 
				
				for (int ctr = 0; ctr < hyperparams_count; ctr++) {
					latent_log_hyperparams[ctr] += step_size*gradient[ctr];
				}
				
				try{
					recompute_k();
					recompute_k_inv();
				}
				catch (Exception e) {
					latent_log_hyperparams = Arrays.copyOf(prev_sane_hyperparams, hyperparams_count);
					recompute_k();
					recompute_k_inv();
					step_size *= decayer;
					continue;
				}
				
				double new_likhood = get_likhood(y_u);
				
				if(Double.isNaN(new_likhood) || Double.isInfinite(new_likhood) || new_likhood <= cur_likhood) {
					latent_log_hyperparams = Arrays.copyOf(prev_sane_hyperparams, hyperparams_count);
					recompute_k();
					recompute_k_inv();
					step_size *= decayer;
					continue;
				}
				else {
					cur_likhood = new_likhood;
					if(cur_likhood > target_likhood) {
						target_likhood = cur_likhood;
						target_hyperparams = Arrays.copyOf(latent_log_hyperparams, hyperparams_count);
					}
					progression = true;
					break;
				}
				
			}
			
		}
		
	}
	
	protected double get_likhood(double[][] y_u) throws Exception {
		
		double complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
		double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y_u), k_inv), y_u)[0][0];
		double constant = -(n/2.0)*Math.log(Math.PI*2);
		
		return complexity_penalty + data_fit + constant;
//		return data_fit;
	}
	
	protected void recompute_k() throws Exception {
		//recomputing covariance matrix and its inverse
		//assuming sliding is already enabled
		
		for(int ctr = 0; slide && ctr < w_size; ctr++) {
			for(int ctr2 = 0; ctr2 < w_size; ctr2++) {
				k[ctr][ctr2] = kernel_func(dp_window[(w_start+ctr) % w_size], dp_window[(w_start+ctr2) % w_size]);
			}
		}
	}
	
	protected void recompute_k_inv() throws Exception {
		k_inv = MatrixOp.fast_invert_psd(k);
	}

	protected double kernel_func(double[][] dp1, double dp2[][]) {
		
		double kernel_measure;
		double sum = 0;
			
		for(int ctr = 0; ctr < dp2.length; ctr++) {
			double dif = dp2[ctr][0] - ((dp1 == null) ? 0 : dp1[ctr][0]); 
			sum += dif*dif*Math.pow(Math.E, -2*latent_log_hyperparams[ctr+2]);
		}
		
		kernel_measure = Math.pow(Math.E, -0.5*sum)*Math.pow(Math.E, 2*latent_log_hyperparams[1]) + ((sum == 0) ? Math.pow(Math.E, 2*latent_log_hyperparams[0]) : 0);
		
		return kernel_measure;
	}
	
//	private void print_marginal_likelihood_vals(double[][] responses_matrix, double current_mlik) throws Exception {
//		
//		double min_sigma_y = 0.1;
//		double max_sigma_y = 2.0;
//		double step_sigma_y = 0.2;
//		
//		double min_sigma_w = 0.5;
//		double max_sigma_w = 5;
//		double step_sigma_w = 0.25;
//				
//		double min_lengthscale = 45;
//		double max_lengthscale = 50;
//		double length_scale_step = 0.5;
//		
//		double step_number = (max_lengthscale - min_lengthscale) / length_scale_step;
//		double total_steps = (int) Math.pow(step_number, feature_count);
//		
//		double[] optimum_hyperparams = new double[feature_count+2];
//		
//		for(int ctr = 0; ctr < feature_count+2; ctr++)
//			optimum_hyperparams[ctr] = hyperparams[ctr];
//		
//		for(int cur_step = 0 ; cur_step < total_steps; cur_step++) {
//			for (double cur_sigma_y = min_sigma_y; cur_sigma_y <= max_sigma_y; cur_sigma_y += step_sigma_y) {
//				for (double cur_sigma_w = min_sigma_w; cur_sigma_w <= max_sigma_w; cur_sigma_w += step_sigma_w) {
//					
//					double temp = cur_step;
//					
//					for(int ctr = feature_count-1; ctr >= 0; ctr--) {
//						double cur_level = Math.pow(step_number, ctr);
//						double cur_number = Math.floor(temp/cur_level); 
//						hyperparams[ctr+2] = cur_number*length_scale_step + min_lengthscale;
//						temp -= cur_number*cur_level;
//					}
//					
//					hyperparams[0] = cur_sigma_y;
//					hyperparams[1] = cur_sigma_w;
//					
//					a = 1/(hyperparams[0]*hyperparams[0]);
//					b = 1/(hyperparams[1]*hyperparams[1]);
//					
//					recompute_k();
//					
//					try {
//						recompute_k_inv();
//					}
//					catch (Exception e) {
//						//e.printStackTrace();
//						continue;
//					}
//					
//					double complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
//					double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(responses_matrix), k_inv), responses_matrix)[0][0];
//					double constant = -(n/2.0)*Math.log(Math.PI*2);
//					
//					double marginal_lhood = complexity_penalty + data_fit + constant;   
//					
//					if(current_mlik < marginal_lhood) {
//						current_mlik = marginal_lhood;
//						
//						System.out.print(hyperparams[0] + ", " + hyperparams[1]);
//						
//						for(int ctr = 0; ctr < feature_count; ctr++)
//								System.out.print(", " + hyperparams[ctr+2]);
//						
//						System.out.println(" -> " + marginal_lhood + " = " + complexity_penalty + " + " + data_fit + " + " + constant);
//						
//						for(int ctr = 0; ctr < feature_count+2; ctr++)
//							optimum_hyperparams[ctr] = hyperparams[ctr];
//						
//					}
//				}
//			}
//		}
//		
//		for(int ctr = 0; ctr < feature_count+2; ctr++)
//			hyperparams[ctr] = optimum_hyperparams[ctr];
//		
//		a = 1/(optimum_hyperparams[0]*optimum_hyperparams[0]);
//		b = 1/(optimum_hyperparams[1]*optimum_hyperparams[1]);
//		
//		recompute_k();
//		recompute_k_inv();
//	}
}
