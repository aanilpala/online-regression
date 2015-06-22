package regression.online.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import regression.online.learner.BayesianMAP;
import regression.online.learner.BayesianMAPForgetting;
import regression.online.learner.BayesianMLE;
import regression.online.learner.BayesianMLEForgetting;
import regression.online.learner.BayesianPredictive;
import regression.online.learner.BayesianMAPWindowed;
import regression.online.learner.BayesianMLEWindowed;
import regression.online.learner.BayesianPredictiveWindowed;
import regression.online.learner.GPWindowedFixedMean;
import regression.online.learner.GPWindowedOLSMean;
import regression.online.learner.GPWindowedZeroMean;
import regression.online.learner.KernelRegression;
import regression.online.learner.Regressor;
import regression.online.model.Prediction;

public class RegressorTest {

	public Regressor reg;
	public List<double[][]> data_points;
	public List<Double> responses;
	public double smse;
	public double interval_containment_rate;
	public double avg_interval_width;
	
	public RegressorTest(Regressor reg, List<double[][]> data_points, List<Double> responses) {
		
		if(data_points.size() != responses.size()) {
			System.out.println("Size Inconsistency");
			return;
		}
		
		this.data_points = data_points;
		this.responses = responses;
		this.reg = reg;
		this.smse = 0;
		
	}
	
	public void test(boolean dump_logs) {
		int interval_miss_count= 0;
		long accumulated_squared_error = 0;
		long accumulated_interval_width = 0;
		int burn_in_number = reg.get_burn_in_number();
		int m_n = 0;
		int interval_width_counter = 0;
		
		double m_new, m_old = 0, s_new = 0, s_old = 0; // needed for target variance computation
		
		FileWriter fw = null;
		if(dump_logs) {
			try {
				fw = new FileWriter("./data/logs/" + this.reg.name + "_logs.txt");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		for(int ctr = 0; ctr < data_points.size(); ctr++) {
			double[][] dp = data_points.get(ctr);
			Double response = responses.get(ctr);
			
			Prediction pred = null;
			try {
				pred = reg.predict(dp);
				reg.update(dp, response, pred);
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
					if(pred.upper_bound < response || response < pred.lower_bound) 
						interval_miss_count++;
					
					accumulated_interval_width += (long) ((pred.upper_bound - pred.lower_bound)*100);
				}
			}
			
			if(dump_logs) {
				try {
					fw.write("target: " + response + " predicted: " + pred.toString());
					fw.write("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	
		double target_variance = s_new/(m_n-1);
		smse = Math.sqrt(accumulated_squared_error/(target_variance*(10000.0)*m_n));
		interval_containment_rate = (1 - (interval_miss_count / (double) (interval_width_counter)))*100;
		avg_interval_width = accumulated_interval_width/(100.0*interval_width_counter);
		
		if(dump_logs) {
			try {
				fw.write("SUMMARY\n");
				fw.write("SMSE = " + smse + " Avg Inteval Width = " + avg_interval_width + " Interval Containment Ratio = " + interval_containment_rate + "%");
				fw.flush();
				fw.close();
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
		return reg.name;
	}
	
	public static void main(String[] args) {
		
		List<double[][]> data_points = new ArrayList<double[][]>();
		List<Double> responses = new ArrayList<Double>();
		
		List<Regressor> regs = new ArrayList<Regressor>();
		List<RegressorTest> reg_testers = new ArrayList<RegressorTest>();
 		
		try(BufferedReader br = new BufferedReader(new FileReader("./data/fakeOpData.csv"))) {
		    for(String line; (line = br.readLine()) != null; ) {
		    	
		    	String[] tokens = line.split("\\|");
		    	String[] features_tokens = tokens[0].split("\t");
		    	
		    	double[][] dp = new double[features_tokens.length][1];
		    	
		    	for(int ctr = 0; ctr < features_tokens.length; ctr++) {
		    		dp[ctr][0] = Double.parseDouble(features_tokens[ctr].trim());
		    	}
		    	
		    	data_points.add(dp);
		    	
		    	String[] response_tokens = tokens[1].split("\t");
		    	double response = Double.parseDouble(response_tokens[response_tokens.length-1].trim()); // in ms
		    	responses.add(response);
		    	
		    }
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int input_width = data_points.get(0).length;
		
//		regs.add(new BayesianMLE(input_width, false));
//		regs.add(new BayesianMLE(input_width, true));
//		regs.add(new BayesianMLEWindowed(input_width, false));
//		regs.add(new BayesianMLEWindowed(input_width, true));
//		regs.add(new BayesianMLEForgetting(input_width, false));
//		regs.add(new BayesianMLEForgetting(input_width, true));
//		
//		regs.add(new BayesianMAP(input_width, false, 0.1, 2));
//		regs.add(new BayesianMAP(input_width, true, 0.1, 2));
//		regs.add(new BayesianMAPWindowed(input_width, false, 0.1, 2));
//		regs.add(new BayesianMAPWindowed(input_width, true, 0.1, 2));
//		regs.add(new BayesianMAPForgetting(input_width, false, 0.1, 2));
//		regs.add(new BayesianMAPForgetting(input_width, true, 0.1, 2));
		
//		regs.add(new BayesianPredictive(input_width, false, 0.1, 2));
//		regs.add(new BayesianPredictive(input_width, true, 0.1, 2));
//		regs.add(new BayesianPredictiveWindowed(input_width, false, 0.1, 2));
//		regs.add(new BayesianPredictiveWindowed(input_width, true, 0.1, 2));
		
//		regs.add(new NadaryaWatsonEstimator(input_width));
		
//		regs.add(new KernelRegression(input_width, 100));
		
		regs.add(new GPWindowedFixedMean(input_width, 50, 0, 2));
//		
//		regs.add(new GPWindowedOLSMean(input_width, 50, 0, 5));
//		
//		regs.add(new GPWindowedZeroMean(input_width, 50, 0, 2));
		
		
		for(Regressor each : regs)
			reg_testers.add(new RegressorTest(each, data_points, responses));

		for(RegressorTest each : reg_testers) {
			each.test(true);
			System.out.println(each.getRegName() + " SMSE= " + each.getSMSE() + " Avg Inteval Width = " + each.avg_interval_width + " IContainment: " + each.getContainmentRate() + "%");
		}
		
	}
}
