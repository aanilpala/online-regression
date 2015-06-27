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
import regression.online.learner.GPWindowedFixedMean;
import regression.online.learner.GPWindowedOLSMean;
import regression.online.learner.GPWindowedZeroMean;
import regression.online.learner.KernelRegression;
import regression.online.learner.Regressor;
import regression.online.model.Prediction;

public class RegressorTest {

	public Regressor reg;
	
	List<double[][]> data_points;
	List<Double> responses;
	public String ds_name;
	public double smse, rmse;
	public double interval_containment_rate;
	public double avg_interval_width;
	public double standardized_mean_interval_width;
	public double avg_prediction_time;
	public double avg_update_time;
	public double avg_tuning_time;
	
	public RegressorTest(Regressor reg, TestCase testcase) {
		
		this.data_points = testcase.data_points;
		this.responses = testcase.responses;
		
		if(data_points.size() != responses.size()) {
			System.out.println("Size Inconsistency");
			return;
		}
		
		this.ds_name = testcase.data_set_name;
		this.reg = reg;
		this.smse = 0;
		this.rmse = 0;
		
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
		long accumulated_target = 0;
		
		double m_new, m_old = 0, s_new = 0, s_old = 0; // needed for target variance computation
		
		FileWriter fw = null;
		FileWriter fw2 = null;
		if(dump_logs) {
			try {
				fw = new FileWriter("./data/logs/" + this.reg.get_name() + "_on_" + this.ds_name + "_logs.txt");
				fw2 = new FileWriter("./data/plotdata/" + this.reg.get_name() + "_on_" + this.ds_name + "_plotdata.txt");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		for(int ctr = 0; ctr < data_points.size(); ctr++) {
			double[][] dp = data_points.get(ctr);
			Double response = responses.get(ctr);
			
			Prediction pred = null;
			try {
				long startTime = System.nanoTime();    
				pred = reg.predict(dp);
				aggregate_prediction_time += System.nanoTime() - startTime;
				
				startTime = System.nanoTime();
				reg.update(dp, response, pred);
				long elapsedTime = System.nanoTime() - startTime;
				
				if(reg.is_tuning_time()) {
					aggregate_tuning_time += elapsedTime;
					optimizer_run_count++;
				}
				else aggregate_update_time += elapsedTime;
				
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
						m_new = m_old + (response - m_old)/m_n;
						s_new = s_old + (response - m_old)*(response - m_new);
						
						//for the next iteration
						m_old = m_new;
						s_old = s_new;
					}
					accumulated_squared_error += Math.pow((pred.point_prediction - response)*100, 2);
				}
				if(!Double.isNaN(pred.lower_bound) && !(Double.isNaN(pred.lower_bound))) {
					interval_width_counter++;
					accumulated_target += response*1000000;
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
					plot_data_row += response;
					
					fw.write( "target: " + response + " predicted: " + pred.toString());
					fw.write("\n");
					
					fw2.write(plot_data_row);
					fw2.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	
		
		// computing stats
		double target_variance = s_new/(m_n-1);
		double residual_variance = accumulated_squared_error/(10000.0*m_n);
		
//		System.out.println(target_variance);
//		System.out.println(residual_variance);
		
		smse = residual_variance/target_variance;
		rmse = Math.sqrt(residual_variance);
		interval_containment_rate = (1 - (interval_miss_count / (double) (interval_width_counter)))*100;
		avg_interval_width = accumulated_interval_width/(100.0*interval_width_counter);
		double target_mean = accumulated_target/1000000.0;
		standardized_mean_interval_width = avg_interval_width/target_mean;
		
		// computing time stats (in ms)
		avg_prediction_time = (aggregate_prediction_time / (long) data_points.size()) / 1000000.0;
		avg_update_time = (aggregate_update_time / (long) ((data_points.size() - optimizer_run_count))) / 1000000.0;
		avg_tuning_time = optimizer_run_count != 0 ? (aggregate_tuning_time / (long) optimizer_run_count) / 1000000.0 : Double.NaN;; 
		
		
		if(dump_logs) {
			try {
				fw.write("SUMMARY\n");
				fw.write("SMSE = " + smse + " RMSE = " + rmse + " AIW = " + avg_interval_width + " SMIW = " + standardized_mean_interval_width + " ICR = " + interval_containment_rate + "%\n");
				fw.write("Avg Prediction Time : " + avg_prediction_time + "msec\n");
				fw.write("Avg Update Time : " + avg_update_time + "msec\n");
				fw.write("Avg Optimzation Time : " + avg_tuning_time + "msec\n");
				fw.flush();
				fw.close();
				fw2.flush();
				fw2.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public double getSMSE(){
		return smse;
	}
	
	public double getContainmentRate(){
		return interval_containment_rate;
	}
	
	public String getRegName() {
		return reg.get_name();
	}
	
	public static void main(String[] args) {
		
		List<TestCase> test_cases = new ArrayList<TestCase>();
		
		File data_path = new File("./data/input"); 
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
		
		FileWriter fw = null;
		FileWriter fw2 = null;
		
		try {
			fw = new FileWriter("./data/" + "comparison_table.csv");
			fw.write("NAME\t" + "TEST\t" + "SMSE\t" + "RMSE\t" + "AIW\t" + "SMIW\t" + "ICR\t" + "PRED_TIME\t" + "UPDATE_TIME\t" + "TUNING_TIME\n");
			
			fw2 = new FileWriter("./data/" + "agg_comparison_table.csv");
			fw2.write("NAME\t" + "SMSE\t" + "RMSE\t" + "AIW\t" + "SMIW\t" + "ICR\t" + "PRED_TIME\t" + "UPDATE_TIME\t" + "TUNING_TIME\n");
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		HashMap<Integer, List<RegressorTest>> regtests = new HashMap<Integer, List<RegressorTest>>();
		for(TestCase testcase : test_cases) {
			List<Regressor> regs = new ArrayList<Regressor>();
			
			int input_width = testcase.input_width;
			
			regs.add(new BayesianMLEWindowed(input_width, 50));	// OLS asymptotic prediction intervals
			regs.add(new BayesianMLEWindowedMapped(input_width, 50));	// OLS asymptotic prediction intervals
			regs.add(new BayesianMLEForgetting(input_width));	// Ad-Hoc prediction intervals
			regs.add(new BayesianMLEForgettingMapped(input_width));		// Ad-Hoc prediction intervals
			
			regs.add(new BayesianMAPWindowed(input_width, 50, 0.1, 2));	// OLS asymptotic prediction intervals
			regs.add(new BayesianMAPWindowedMapped(input_width, 50, 0.1, 2));	// OLS asymptotic prediction intervals
			regs.add(new BayesianMAPForgetting(input_width, 0.1, 2));	// Ad-Hoc prediction intervals
			regs.add(new BayesianMAPForgettingMapped(input_width, 0.1, 2));		// Ad-Hoc prediction intervals

			regs.add(new BayesianPredictive(input_width, 0.1, 2));	// Gaussian Posterior Variance as the prediction intervals
			regs.add(new BayesianPredictiveMapped(input_width, 0.1, 2));	// Gaussian Posterior Variance as the prediction intervals
			regs.add(new BayesianPredictiveWindowed(input_width, 50, 0.1, 2));	// Gaussian Posterior Variance as the prediction intervals + sigma_w tuning
			regs.add(new BayesianPredictiveWindowedMapped(input_width, 50, 0.1, 2));	// Gaussian Posterior Variance as the prediction intervals + sigma_w tuning
			
			regs.add(new KernelRegression(input_width, 50));	// Confidence Intervals instead of Prediction Intervals
			
			regs.add(new GPWindowedFixedMean(input_width, 50, 0, 2, false));		
			regs.add(new GPWindowedOLSMean(input_width, 50, 0, 2, false));		// OLS models the data as much as it can and the residuals are modeled by the GP Regression
			regs.add(new GPWindowedZeroMean(input_width, 50, 0, 2, false));
			
			for(Regressor each : regs) {
				int cur_id = each.getId();
				List<RegressorTest> cur_tests = regtests.get(cur_id);
				if(cur_tests == null) {
					cur_tests = new ArrayList<RegressorTest>();
					regtests.put(cur_id, cur_tests);
				}
				
				cur_tests.add(new RegressorTest(each, testcase));
			}			
		}

		for(Integer key : regtests.keySet()) {
			double avg_smse = 0;
			double avg_rmse = 0;
			double avg_aiw = 0;
			double avg_smiw = 0;
			double avg_icr = 0;
			
			double avg_pt = 0;
			double avg_ut = 0;
			double avg_tt = 0;
			
			List<RegressorTest> cur_tests = regtests.get(key);
			
			String cur_reg = cur_tests.get(0).getRegName();
			
			for(RegressorTest each : cur_tests) {
				each.test(true);
				System.out.println(each.reg.get_name() + " " + "on dataset " + each.ds_name + " performance summary");
				System.out.println("SMSE = " + each.smse + " RMSE = " + each.rmse + " AIW = " + each.avg_interval_width + " SMIW = " + each.standardized_mean_interval_width + " ICR = " + each.interval_containment_rate + "%");
				try {
					fw.write(each.getRegName() + "\t" 
							+ each.ds_name + "\t" 
							+ each.smse + "\t" 
							+ each.rmse + "\t" 
							+ each.avg_interval_width + "\t" 
							+ each.standardized_mean_interval_width + "\t" 
							+ each.interval_containment_rate + "\t"
							+ each.avg_prediction_time + "\t"
							+ each.avg_update_time + "\t"
							+ each.avg_tuning_time);
							
					fw.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
				System.out.println("Avg Prediction Time : " + each.avg_prediction_time + " msec");
				System.out.println("Avg Update Time : " + each.avg_update_time + " msec");
				System.out.println("Avg Tuning Time : " + each.avg_tuning_time + " msec\n");
				
				avg_smse += each.smse;
				avg_rmse += each.rmse;
				avg_aiw += each.avg_interval_width;
				avg_smiw += each.standardized_mean_interval_width;
				avg_icr += each.interval_containment_rate;
				
				avg_pt += each.avg_prediction_time;
				avg_ut += each.avg_update_time;
				avg_tt += each.avg_tuning_time;
			}
			
			int test_count = cur_tests.size();
			
			avg_rmse /= test_count;
			avg_smse /= test_count;
			avg_aiw /= test_count;
			avg_smiw /= test_count;
			avg_icr /= test_count;
			
			avg_pt /= test_count;
			avg_ut /= test_count;
			avg_tt /= test_count;
			
			try {
				fw2.write(cur_reg + "\t" 
						+ avg_rmse + "\t" 
						+ avg_smse + "\t" 
						+ avg_aiw + "\t" 
						+ avg_smiw + "\t" 
						+ avg_icr + "\t"
						+ avg_pt + "\t"
						+ avg_ut + "\t"
						+ avg_tt);
						
				fw2.write("\n");
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
		
		try {
			fw.flush();
			fw.close();
			
			fw2.flush();
			fw2.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
