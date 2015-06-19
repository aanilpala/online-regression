package regression.online.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class SynthDataGenerator {

	Random rand;
	int feature_count;
	int instance_count;
	BufferedWriter writer;
	double[] params;
	double[] exp_vars;
	double target;
	double noise_var = 0.001;
	
	public SynthDataGenerator(BufferedWriter writer, int feature_count, int instance_count) {
		
		rand = new Random();
		this.feature_count = feature_count;
		this.writer = writer;
		this.instance_count = instance_count;
		
		params = new double[feature_count];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			int sign = 1; //rand.nextBoolean() ? 1 : -1;
			params[ctr] = rand.nextDouble()*10*sign;
			System.out.println(params[ctr]);
		}
		
		exp_vars = new double[feature_count];
		
		
	}
	
	public void generate() throws IOException {
		for(int ctr = 0; ctr < instance_count; ctr++) {
			
			target = 0;
			
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				
				int sign = 1; //rand.nextBoolean() ? 1 : -1;
				exp_vars[ctr2] = sign*rand.nextDouble()*10;
				writer.append(((Double) exp_vars[ctr2]).toString() + "\t");
				
				
				target += exp_vars[ctr2]*params[ctr2] + exp_vars[ctr2]*12 + Math.pow(Math.E, exp_vars[ctr2]);
//				target += exp_vars[ctr2]*exp_vars[ctr2]*exp_vars[ctr2]*params[ctr2] + Math.log(exp_vars[ctr2]);
//				target += exp_vars[ctr2]*params[ctr2];
			}
			
			if(rand.nextBoolean()) target += rand.nextGaussian()*noise_var;
			else target -= rand.nextGaussian()*noise_var;
			
			writer.append("|");
			writer.append(((Double) target).toString());
			writer.append("\n");
			
		}
		
		writer.flush();
		writer.close();
	}
	
	public static void main(String[] args) {
		
		
		BufferedWriter writer = null;
		try {
			writer = new BufferedWriter(new FileWriter(new File("./data/fakeOpData.csv"), false));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		SynthDataGenerator sgen = new SynthDataGenerator(writer, 2, 2500);
		try {
			sgen.generate();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	
}
