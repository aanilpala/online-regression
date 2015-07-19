package regression.offline.learner;

import java.util.Arrays;
import java.util.Random;

import regression.util.MatrixOp;

public class GPRegressionBatch extends BatchRegressor {

	double[][] k; // gram matrix
	double[][] k_inv; // inverse gram matrix
	
	double[][][] training_set;
	double[][] responses;
	
	// hyperparams
	double[] latent_log_hyperparams; // length-scale array;
	int hyperparams_count;
	
	double init_lengthscale = 1;
		
	Random rand = new Random();
	boolean sigma_y_is_inhibated;
	
	public GPRegressionBatch(int input_width, int training_set_size, boolean sigma_y_is_inhibited) {
		super(input_width, training_set_size, sigma_y_is_inhibited);
		
		this.sigma_y_is_inhibated = sigma_y_is_inhibited;
		
		training_set = new double[training_set_size][input_width][1];
		responses = new double[training_set_size][1];
		
		k = new double[training_set_size][training_set_size]; 
		k_inv = new double[training_set_size][training_set_size];
		
		hyperparams_count = feature_count + 2;
		latent_log_hyperparams = new double[hyperparams_count];
		
		latent_log_hyperparams[0] = Math.log(0.1);
		latent_log_hyperparams[1] = Math.log(2);
		
		for(int ctr = 2; ctr < hyperparams_count; ctr++) {
			latent_log_hyperparams[ctr] = Math.log(init_lengthscale);
		}		
		
		double val = Math.pow(Math.E, 2*latent_log_hyperparams[0]) + Math.pow(Math.E, 2*latent_log_hyperparams[1]);
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
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
	
	@Override
	public void train(double[][][] dps, double[][] targets) throws Exception {
		
		compute_means(dps, targets);
		determine_scaling_factors();
		scale_training_data(dps, targets);
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				training_set[ctr][ctr2][0] = dps[ctr][ctr2][0];
			}
		}
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			responses[ctr][0] = targets[ctr][0];
		}
		
		recompute_k();
		recompute_k_inv();
		
		double marginal_lhood = get_likhood(responses);
		
		double[] gradient = new double[latent_log_hyperparams.length];
		
		set_gradients(gradient, responses);
		
		optimize_hyperparams(responses, marginal_lhood);
		
	}
	
	@Override
	public double predict(double[][] dp) throws Exception {
		
		dp = scale_input(dp);
		
		double[][] spare_column = new double[training_set_size][1];
		
		for(int ctr = 0; ctr < training_set_size; ctr++)
			spare_column[ctr][0] = gaussian_kernel(training_set[ctr], dp);
		
		double[][] temp_column = MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv);
		
		double y = 0;
		
		for(int ctr = 0; ctr < training_set_size; ctr++)
			y += temp_column[0][ctr]*(responses[ctr][0]); 
		
		return target_postscaler(y);
	}
	
	
	protected void set_gradients(double[] gradient, double[][] y) throws Exception {
		
		double[][] derivative_cov_matrix = new double[training_set_size][training_set_size];
		
		if(!sigma_y_is_inhibated ) {
			// gradient log sigma_y
				for(int ctr = 0; ctr < training_set_size; ctr++) {
					for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
						if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2.0*Math.pow(Math.E, 2*latent_log_hyperparams[0]); 
						else derivative_cov_matrix[ctr][ctr2] = 0;
					}
				}
			gradient[0] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
		}
		else gradient[0] = 0;
				
		// gradient log sigma_w
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
				if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 2.0*Math.pow(Math.E, 2*latent_log_hyperparams[1]); //2.0*hyperparams[1]; 
				else derivative_cov_matrix[ctr][ctr2] = (2.0)*k[ctr][ctr2]; // (2.0/hyperparams[1])*k[ctr][ctr2];
			}
		}
					
		gradient[1] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
					
		// lengthscale gradients
		for(int ctr3 = 2; ctr3 < hyperparams_count; ctr3++) {
			for(int ctr = 0; ctr < training_set_size; ctr++) {
				for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
					if(ctr == ctr2) derivative_cov_matrix[ctr][ctr2] = 0;
					else {
						double dif = training_set[ctr][ctr3-2][0] - training_set[ctr2][ctr3-2][0];
						derivative_cov_matrix[ctr][ctr2] = dif*dif*Math.pow(Math.E, -2*latent_log_hyperparams[ctr3])*k[ctr][ctr2]; //(((dif*dif))/(Math.pow(hyperparams[ctr3], 3)))*k[ctr][ctr2];
					}
				}
			}
			gradient[ctr3] = -0.5*MatrixOp.trace(MatrixOp.mult(k_inv, derivative_cov_matrix)) + 0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y), k_inv), derivative_cov_matrix), k_inv), y)[0][0];
		}
		
		
	}
	
	protected void optimize_hyperparams(double[][] y_u, double init_likhood) throws Exception {
		
		// optimizer parameters
		int iteration_counter = 0;
		int max_it = 10;
		double decayer = 0.1;
		int decay_counter;
		int max_decay_count = 10;
		double step_size;
		double max_approx_optima_gradient = 10E-20;
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
			
			iteration_counter++;
			if(progression && iteration_counter > max_it) {
				cur_likhood = target_likhood;
				latent_log_hyperparams = Arrays.copyOf(target_hyperparams, hyperparams_count);
				recompute_k();
				recompute_k_inv();
				break;
			}
			if(!progression) {
				if(reset_counter >= reset_limit) {
					cur_likhood = target_likhood;
					latent_log_hyperparams = Arrays.copyOf(target_hyperparams, hyperparams_count);
					recompute_k();
					recompute_k_inv();
					break;
				}
				else {
					if(verbouse) System.out.println("Resetting... target_likhood: " + target_likhood);
					iteration_counter = 0;
					while(true) {
						// pick random values
						for(int ctr = sigma_y_is_inhibated ? 1 : 0; ctr < hyperparams_count; ctr++)
							latent_log_hyperparams[ctr] = Math.log(10*rand.nextDouble());
						
						boolean feasible = true;
						for (int ctr = sigma_y_is_inhibated ? 1 : 0; ctr < hyperparams_count; ctr++) {
							double test = Math.pow(Math.E, 2*latent_log_hyperparams[ctr]);
							if(Double.isInfinite(test) || Double.isNaN(test)) {
								feasible = false;
								break;
							}
						}
						
						// early rollback -before recomputing the covariance matrix and its inverse
						if(!feasible) {
							//latent_log_hyperparams = Arrays.copyOf(prev_sane_hyperparams, hyperparams_count);
							continue;
						}
						
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
			for (int ctr = sigma_y_is_inhibated ? 1 : 0; ctr < gradient.length; ctr++) {
				if(Math.abs(gradient[ctr]) > max_approx_optima_gradient) {
					optima_found = false;
					break;
				}
			}
			
			if(optima_found && cur_likhood >= target_likhood) break;
			
//			step_size = default_step_size;
			
			step_size = adapt_step_size(gradient);
			decay_counter = 0;
			
			progression = false;
			
			while(true) {
				
				if(decay_counter++ > max_decay_count) break;
				
				prev_sane_hyperparams = Arrays.copyOf(latent_log_hyperparams, hyperparams_count); 
				
				for (int ctr = sigma_y_is_inhibated ? 1 : 0; ctr < hyperparams_count; ctr++) {
					latent_log_hyperparams[ctr] += step_size*gradient[ctr];
				}
				
				boolean feasible = true;
				for (int ctr = sigma_y_is_inhibated ? 1 : 0; ctr < hyperparams_count; ctr++) {
					double test = Math.pow(Math.E, 2*latent_log_hyperparams[ctr]);
					if(Double.isInfinite(test) || Double.isNaN(test)) {
						step_size *= decayer;
						feasible = false;
						break;
					}
				}
				
				// early rollback -before recomputing the covariance matrix and its inverse
				if(!feasible) {
					latent_log_hyperparams = Arrays.copyOf(prev_sane_hyperparams, hyperparams_count);
					step_size *= decayer;
					continue;
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
						prev_sane_hyperparams = Arrays.copyOf(latent_log_hyperparams, hyperparams_count);
					}
					progression = true;
					break;
				}
				
			}
			
		}
		
	}
	
	private double adapt_step_size(double[] gradient) {
		double max_log = Double.NEGATIVE_INFINITY;
		
		for(double component : gradient) {
			double cur_log = Math.log10(Math.abs(component));
			if(cur_log > max_log) max_log = cur_log;
		}
		
		
		return Math.pow(10.0, -1*max_log);
	}

	protected double get_likhood(double[][] y_u) throws Exception {
		
		double complexity_penalty = -0.5*Math.log(MatrixOp.get_det_of_psd_matrix(k));
		double data_fit = -0.5*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(y_u), k_inv), y_u)[0][0];
		double constant = -(training_set_size/2.0)*Math.log(Math.PI*2);
		
		return complexity_penalty + data_fit + constant;
//		return data_fit;
	}
	
	protected void recompute_k() throws Exception {
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
				k[ctr][ctr2] = gaussian_kernel(training_set[ctr], training_set[ctr2]);
			}
		}
	}
	
	protected void recompute_k_inv() throws Exception {
		k_inv = MatrixOp.fast_invert_psd(k);
	}

	protected double gaussian_kernel(double[][] dp1, double dp2[][]) {
		
		double kernel_measure;
		double sum = 0;
			
		for(int ctr = 0; ctr < dp2.length; ctr++) {
			double dif = dp2[ctr][0] - ((dp1 == null) ? 0 : dp1[ctr][0]); 
			sum += dif*dif*Math.pow(Math.E, -2*latent_log_hyperparams[ctr+2]);
		}
		
		double exp = Math.pow(Math.E, -0.5*sum);
		if(sum > 0 && Double.isNaN(exp)) {
			System.out.println("ILLEGAL KERNEL");
		}
		
		kernel_measure = exp*Math.pow(Math.E, 2*latent_log_hyperparams[1]) + ((sum == 0) ? Math.pow(Math.E, 2*latent_log_hyperparams[0]) : 0);
		
		return kernel_measure;
	}
	
	
}
