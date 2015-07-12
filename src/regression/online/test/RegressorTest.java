package regression.online.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import aima.core.learning.framework.DataSet;
import regression.online.learner.BayesianMAPForgetting;
import regression.online.learner.BayesianMAPForgettingMapped;
import regression.online.learner.BayesianMAPWindowedMapped;
import regression.online.learner.BayesianMLEForgetting;
import regression.online.learner.BayesianMLEForgettingMapped;
import regression.online.learner.BayesianMLEWindowedMapped;
import regression.online.learner.BayesianPredictive;
import regression.online.learner.BayesianPredictiveMapped;
import regression.online.learner.BayesianMAPWindowed;
import regression.online.learner.BayesianMLEWindowed;
import regression.online.learner.BayesianPredictiveWindowed;
import regression.online.learner.BayesianPredictiveWindowedMapped;
import regression.online.learner.GPWindowedGaussianKernelAvgMean;
import regression.online.learner.GPWindowedGaussianKernelOLSMean;
import regression.online.learner.GPWindowedGaussianKernelZeroMean;
import regression.online.learner.KernelRegression;
import regression.online.learner.Regressor;
import regression.online.model.Prediction;

public class RegressorTest {

	public Regressor reg;
	
	List<double[][]> data_points;
	List<Double> responses;
	public String ds_name;
	
	public double rmse, smse;
	public double ict;
	public double aiw, saiw;
	public double apt, hpt;
	public double aut, hut;
	public double att, htt;
	
	public RegressorTest(Regressor reg, TestCase testcase) {
		
		this.data_points = testcase.data_points;
		this.responses = testcase.responses;
		
		if(data_points.size() != responses.size()) {
			System.out.println("Size Inconsistency");
			return;
		}
		
		this.ds_name = testcase.data_set_name;
		this.reg = reg;
		
		// initializing the stats
		this.smse = 0;
		this.rmse = 0;
		
		this.hpt = 0;
		this.hut = 0;
		this.htt = 0;
		
	}
	
	public void test(boolean dump_logs) {
		int interval_miss_count= 0;
		long accumulated_squared_error = 0;
		long accumulated_interval_width = 0;
		int burn_in_number = reg.get_burn_in_number();
		int m_n = 0;
		int interval_width_counter = 0;
		long aggregate_update_time = 0;
		long aggregate_tuning_time = 0;
		long aggregate_prediction_time = 0;
		int optimizer_run_count = 0;
		
		double m_new, m_old = 0, s_new = 0, s_old = 0; // needed for target variance computation
		
		FileWriter logs_fw = null;
		FileWriter prediction_plot_fw = null;
		
		if(dump_logs) {
			try {
				logs_fw = new FileWriter(Path.logs_path + this.reg.get_name() + "_on_" + this.ds_name + "_logs.txt");
				prediction_plot_fw = new FileWriter(Path.plots_path + this.reg.get_name() + "_on_" + this.ds_name + "_prediction_plot_data.txt");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		for(int ctr = 0; ctr < data_points.size(); ctr++) {
			double[][] dp = data_points.get(ctr);
			Double response = responses.get(ctr);
			
			Prediction pred = null;
			Double err = Double.NaN;
			
			try {
				long startTime = System.nanoTime();    
				pred = reg.predict(dp);
				long elapsedTime = System.nanoTime() - startTime;
				
				if(elapsedTime/1000000.0 > this.hpt) this.hpt = elapsedTime/1000000.0;
				aggregate_prediction_time += elapsedTime;
				
				startTime = System.nanoTime();
				reg.update(dp, response, pred);
				elapsedTime = System.nanoTime() - startTime;
				
				if(reg.was_just_tuned()) {
					if(elapsedTime/1000000.0 > this.htt) this.htt = elapsedTime/1000000.0;
					aggregate_tuning_time += elapsedTime;
					optimizer_run_count++;
				}
				else {
					if(elapsedTime/1000000.0 > this.hut) this.hut = elapsedTime/1000000.0;
					aggregate_update_time += elapsedTime;
				}
				
			} 
			catch (Exception e1) {
				e1.printStackTrace();
			}
			
			if(ctr >= burn_in_number) {
				if(!Double.isNaN(pred.point_prediction)) {
					m_n++;
					if(m_n == 1) {
						m_old = m_new = response;
						s_old = 0;
					}
					else {
						m_new = m_old + (response - m_old)/((float) m_n);
						s_new = s_old + (response - m_old)*(response - m_new);
						
						//for the next iteration
						m_old = m_new;
						s_old = s_new;
					}
					err = Math.abs(pred.point_prediction - response);
					accumulated_squared_error += Math.pow((pred.point_prediction - response)*100, 2);
				}
				if(!Double.isNaN(pred.lower_bound) && !(Double.isNaN(pred.lower_bound))) {
					interval_width_counter++;
					if(pred.upper_bound < response || response < pred.lower_bound) 
						interval_miss_count++;
					
					accumulated_interval_width += (long) ((pred.upper_bound - pred.lower_bound)*100);
				}
			}
			
			if(dump_logs) {
				try {
					String plot_data_row = "";
					for(int ctr2 = 0; ctr2 < dp.length; ctr2++)
						plot_data_row += dp[ctr2][0] + "\t";
					
					plot_data_row += pred.lower_bound + "\t" + pred.point_prediction + "\t" + pred.upper_bound + "\t";
					plot_data_row += response + "\t" + err;
					
					logs_fw.write( "target: " + response + " predicted: " + pred.toString());
					logs_fw.write("\n");
					
					prediction_plot_fw.write(plot_data_row);
					prediction_plot_fw.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	
		
		// computing stats
		double target_variance = s_new/(float) (m_n-1);
		double residual_variance = (accumulated_squared_error / (long) m_n) / 10000.0;
		
//		System.out.println(target_variance);
//		System.out.println(residual_variance);
		
		smse = residual_variance/target_variance;
		rmse = Math.sqrt(residual_variance);
		ict = (1 - (interval_miss_count / (double) (interval_width_counter)))*100;
		aiw = (accumulated_interval_width / (long) interval_width_counter) / 100.0;
		saiw = aiw / target_variance;
		
		// computing time stats (in ms)
		this.apt = (aggregate_prediction_time / (long) data_points.size()) / 1000000.0;
		this.aut = (aggregate_update_time / (long) ((data_points.size() - optimizer_run_count))) / 1000000.0;
		this.att = optimizer_run_count != 0 ? (aggregate_tuning_time / (long) optimizer_run_count) / 1000000.0 : Double.NaN;;
		
		if(this.htt == 0) this.htt = Double.NaN;
		
		
		if(dump_logs) {
			try {
				logs_fw.write("SUMMARY\n");
				logs_fw.write("SMSE = " + smse + ", RMSE = " + rmse + "\n");
				logs_fw.write("SAIW = " + saiw + ", AIW = " + aiw + " ICR = " + ict + "%\n");
				logs_fw.write("APT = " + apt + " msec" + ", AUT = " + aut + " msec" + ", ATT = " + att + " msec\n");
				logs_fw.write("HPT = " + hpt + " msec" + ", HUT = " + hut + " msec" + ", HTT = " + htt + " msec\n");
				
				logs_fw.flush();
				logs_fw.close();
				prediction_plot_fw.flush();
				prediction_plot_fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	private static void clean_dirs() {
		
		File logs_dir = new File(Path.logs_path);
		File plots_dir = new File(Path.plots_path);
		
		File[] log_files = logs_dir.listFiles();
		File[] plot_files = plots_dir.listFiles();
		
	    if(log_files != null) {
	        for(File f: log_files)
	            if(!f.isDirectory()) f.delete();
	    }
	    
	    if(plot_files != null) {
	        for(File f: plot_files)
	            if(!f.isDirectory()) f.delete();
	    }
		
	}

	public double getSMSE(){
		return smse;
	}
	
	public double getContainmentRate(){
		return ict;
	}
	
	public String getRegName() {
		return reg.get_name();
	}
	
	public static void main(String[] args) {
		
		clean_dirs();
		
		List<TestCase> test_cases = new ArrayList<TestCase>();
		
		File data_path = new File(Path.data_path); 
		for(File file : data_path.listFiles()) {
			if(file.isFile() && !file.isHidden()) {
				try{
					TestCase testcase = new TestCase(file);
					test_cases.add(testcase);
				}
				catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		
		FileWriter comparison_table_fw = null;
		FileWriter agg_comparison_table_fw = null;
		
		try {
			comparison_table_fw = new FileWriter("./data/" + "comparison_table.csv");
			comparison_table_fw.write("NAME\t" + "TEST\t" + "SMSE\t" + "RMSE\t" + "AIW\t" + "SMIW\t" + "ICR\t" + "APT\t" + "AUT\t" + "ATT\t" + "HPT\t" + "HUT\t" + "HTT\n");
			
			agg_comparison_table_fw = new FileWriter("./data/" + "agg_comparison_table.csv");
			agg_comparison_table_fw.write("NAME\t" + "SMSE\t" + "SAIW\t" + "ICR\t" + "APT\t" + "AUT\t" + "ATT\t" + "HPT\t" + "HUT\t" + "HTT\n");
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		HashMap<String, List<RegressorTest>> regtests = new HashMap<String, List<RegressorTest>>();
		for(TestCase testcase : test_cases) {
			List<Regressor> regs = new ArrayList<Regressor>();
			
			int input_width = testcase.input_width;
			
			int window_sizes[] = new int[]{25,50,100};
			
//			regs.add(new BayesianMLEForgetting(input_width, false));	// Ad-Hoc prediction intervals
//			regs.add(new BayesianMLEForgettingMapped(input_width, false));		// Ad-Hoc prediction intervals
			
//			regs.add(new BayesianMAPForgetting(input_width, 0.1, 2, false));	// Ad-Hoc prediction intervals
//			regs.add(new BayesianMAPForgettingMapped(input_width, 0.1, 2, false));		// Ad-Hoc prediction intervals
			
			for(int w_size : window_sizes) {
//				regs.add(new BayesianMLEWindowed(input_width, 25, -1, false));	// OLS asymptotic prediction intervals
//				regs.add(new BayesianMLEWindowedMapped(input_width, 25, -1, false));	// OLS asymptotic prediction intervals

//				regs.add(new BayesianMAPWindowed(input_width, 50, 0.1, 2, -1, false));	// OLS asymptotic prediction intervals
//				regs.add(new BayesianMAPWindowedMapped(input_width, 50, 0.1, 2, -1, false));	// OLS asymptotic prediction intervals

//				regs.add(new BayesianPredictive(input_width, 0.1, 2));	// Gaussian Posterior Variance as the prediction intervals
//				regs.add(new BayesianPredictiveMapped(input_width, 0.1, 2, 50));	// Gaussian Posterior Variance as the prediction intervals
//				regs.add(new BayesianPredictiveWindowed(input_width, 50, 0.1, 2, 1, false));	// Gaussian Posterior Variance as the prediction intervals + sigma_w tuning
//				regs.add(new BayesianPredictiveWindowedMapped(input_width, 50, 0.1, 2, 1, false));	// Gaussian Posterior Variance as the prediction intervals + sigma_w tuning
							
				regs.add(new KernelRegression(input_width, w_size, 1, false));	// Confidence Intervals instead of Prediction Intervals
//				regs.add(new KernelRegression(input_width, w_size, 0, true));	// Confidence Intervals instead of Prediction Intervals
				
				regs.add(new GPWindowedGaussianKernelAvgMean(input_width, w_size, 0.1, 2, false, 1, false));
//				regs.add(new GPWindowedGaussianKernelAvgMean(input_width, w_size, 0.1, 2, false, 0, true));
				
				regs.add(new GPWindowedGaussianKernelOLSMean(input_width, w_size, 0.1, 2, false, 1, false));		// OLS models the data as much as it can and the residuals are modeled by the GP Regression
//				regs.add(new GPWindowedGaussianKernelOLSMean(input_width, w_size, 0.1, 2, false, -1, true));		// OLS models the data as much as it can and the residuals are modeled by the GP Regression
				
				regs.add(new GPWindowedGaussianKernelZeroMean(input_width, w_size, 0.1, 2, false, 1, false));
//				regs.add(new GPWindowedGaussianKernelZeroMean(input_width, w_size, 0.1, 2, false, 0, true));
			}
			
			for(Regressor each : regs) {
				String cur_id = each.name;
				List<RegressorTest> cur_tests = regtests.get(cur_id);
				if(cur_tests == null) {
					cur_tests = new ArrayList<RegressorTest>();
					regtests.put(cur_id, cur_tests);
				}
				
				cur_tests.add(new RegressorTest(each, testcase));
			}			
		}

		for(String name : regtests.keySet()) {
			double avg_smse = 0;
			double avg_saiw = 0;
			double avg_icr = 0;
			
			double avg_apt = 0;
			double avg_aut = 0;
			double avg_att = 0;
			double avg_hpt = 0;
			double avg_hut = 0;
			double avg_htt = 0;
			
			
			List<RegressorTest> cur_tests = regtests.get(name);
			
			String cur_reg = cur_tests.get(0).getRegName();
			
			for(RegressorTest each : cur_tests) {
				each.test(true);
				try {
					comparison_table_fw.write(each.getRegName() + "\t" 
							+ each.ds_name + "\t" 
							+ each.smse + "\t" 
							+ each.rmse + "\t" 
							+ each.aiw + "\t"
							+ each.saiw + "\t"
							+ each.ict + "\t"
							+ each.apt + "\t"
							+ each.aut + "\t"
							+ each.att + "\t"
							+ each.hpt + "\t"
							+ each.hut + "\t"
							+ each.htt + "\n");
							
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				// Console output
				System.out.println(each.reg.get_name() + " " + "on dataset " + each.ds_name + " performance summary");
				System.out.println("SMSE = " + each.smse + ", RMSE = " + each.rmse);
				System.out.println("SAIW = " + each.saiw + ", AIW = " + each.aiw + " ICR = " + each.ict + "%");
				System.out.println("APT = " + each.apt + " msec" + ", AUT = " + each.aut + " msec" + ", ATT = " + each.att + " msec");
				System.out.println("HPT = " + each.hpt + " msec" + ", HUT = " + each.hut + " msec" + ", HTT = " + each.htt + " msec");
				System.out.println();
				
				avg_smse += each.smse;
				avg_saiw += each.aiw;
				avg_icr += each.ict;
				
				avg_apt += each.apt;
				avg_aut += each.aut;
				avg_att += each.att;
			}
			
			int test_count = cur_tests.size();
			
			avg_smse /= test_count;
			avg_saiw /= test_count;
			avg_icr /= test_count;
			
			avg_apt /= test_count;
			avg_aut /= test_count;
			avg_att /= test_count;
			avg_hpt /= test_count;
			avg_hut /= test_count;
			avg_htt /= test_count;
			
			try {
				agg_comparison_table_fw.write(cur_reg + "\t" 
						+ avg_smse + "\t" 
						+ avg_saiw + "\t"
						+ avg_icr + "\t"
						+ avg_apt + "\t"
						+ avg_aut + "\t"
						+ avg_att + "\t"
						+ avg_hpt + "\t"
						+ avg_hut + "\t"
						+ avg_htt + "\n");
						
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
		
		try {
			comparison_table_fw.flush();
			comparison_table_fw.close();
			
			agg_comparison_table_fw.flush();
			agg_comparison_table_fw.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
