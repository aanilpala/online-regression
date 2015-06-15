package regression.online.learner;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import regression.online.model.Prediction;
import regression.online.util.CholeskyDecom;
import regression.online.util.MatrixOp;
import regression.online.util.MatrixPrinter;


public class GPWindowedOLSMean extends GPWindowedBase {
	
	double[] mean_responses; //mean_response;
	double[][] coeff_u; // coefficients of the mean function;
	
	public GPWindowedOLSMean(int input_width, double signal_stddev, double weight_stddev) {
		super(false, input_width);
		
		name = this.getClass().getName();
		
		k = new double[w_size][w_size]; 
		k_inv = new double[w_size][w_size];
		
		mean_responses = new double[w_size];
		coeff_u = new double[3*input_width][1];
		
		//for(int ctr = 0; ctr < input_width; ctr++)
			//coeff_u[ctr][0] = 0; // initially zero mean func
		
		hyperparams = new double[2+input_width];
		
//		hyperparams[0] = signal_stddev;
//		hyperparams[1] = weight_stddev;
		
		hyperparams[0] = rand.nextDouble()*sigma_y_max;
		hyperparams[1] = rand.nextDouble()*sigma_w_max;
		
		for(int ctr = 0; ctr < input_width; ctr++) {
			hyperparams[2+ctr] = rand.nextDouble()*length_scale_max;
		}
		
		a = 1/(hyperparams[0]*hyperparams[0]);
		b = 1/(hyperparams[1]*hyperparams[1]);
		
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
		
		double y = mean_func(dp);
		
		for(int ctr = 0; ctr < n; ctr++) {
			y += temp_column[0][w_size - n + ctr]*(responses[(w_start + ctr) % w_size][0] - mean_responses[(w_start + ctr) % w_size]);
		}
		
		System.out.println(mean_func(dp) + " + " + (y - mean_func(dp)));
		
		double dif = kernel_func(dp, dp) -  MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(spare_column), k_inv), spare_column)[0][0];
		double predictive_deviance = 0;
		
		if(Math.abs(dif) < 0.0000001) predictive_deviance = 0;
		else predictive_deviance = Math.sqrt(dif);
		
		return new Prediction(y, predictive_deviance);
	}
	
	@Override
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		int index = getIndexForDp(dp);
		
		if(index != -1) {
			// reject the update
			// avg the response for the duplicate point
			
			responses[index][0] = (y + responses[index][0])/2.0; 
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
		mean_responses[w_end] = mean_func(dp);
		
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
		
		if(slide && ((update_count - w_size) % hyper_param_update_freq == 0)) { 
			
			count_dps_in_window();
			
			double[][] responses_minus_mean_vector = new double[w_size][1];
			double[][] responses_vector = new double[w_size][1];
			
			// creating response-mean vector
			for(int ctr = 0; ctr < n; ctr++) 
				responses_vector[ctr][0] = responses[(w_start + ctr) % w_size][0];
			
			fit_linear_mean_model(responses_vector);
			
			for(int ctr = 0; ctr < w_size; ctr++)
				mean_responses[ctr] = mean_func(dp_window[(w_start + ctr) % w_size]);
			
			for(int ctr = 0; ctr < w_size; ctr++)
				responses_minus_mean_vector[ctr][0] = responses_vector[ctr][0] - mean_responses[ctr];
			
			double marginal_lhood = get_likhood(responses_minus_mean_vector);
			
			System.out.println("Pre-Optimization Hyperparams :");
			
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				System.out.println("parameter " + ctr + " " + hyperparams[ctr]);
			}
			
			System.out.println("Pre-Optimization Likelihood : " + marginal_lhood);
			
			update_hyperparams_steepestasc(responses_minus_mean_vector, marginal_lhood);
			//update_hyperparams_rprop(responses_minus_mean_vector, marginal_lhood);
			
			marginal_lhood = get_likhood(responses_minus_mean_vector);
			
			System.out.println("Post-Optimization Hyperparams :");
			
			for(int ctr = 0; ctr < hyperparams.length; ctr++) {
				System.out.println("parameter " + ctr + " " + hyperparams[ctr]);
			}
			
			System.out.println("Post-Optimization Likelihood : " + marginal_lhood);
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
		
		coeff_u = MatrixOp.mult(MatrixOp.fast_invert_psd(MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix))), MatrixOp.mult(design_matrix, responses_vector));
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

}
