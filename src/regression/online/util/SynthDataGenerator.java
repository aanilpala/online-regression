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
	double noise_var = 1.0;
	
	public SynthDataGenerator(BufferedWriter writer, int feature_count, int instance_count) {
		
		rand = new Random();
		this.feature_count = feature_count;
		this.writer = writer;
		this.instance_count = instance_count;
		
		params = new double[feature_count];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			params[ctr] = rand.nextDouble()*100;
		}
		
		exp_vars = new double[feature_count];
		
		
	}
	
	public void generate() throws IOException {
		for(int ctr = 0; ctr < instance_count; ctr++) {
			
			target = 0;
			
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				
				exp_vars[ctr2] = Math.abs(rand.nextDouble()*100);
				writer.append(((Double) exp_vars[ctr2]).toString() + "\t");
				
				
				target += exp_vars[ctr2]*Math.log(exp_vars[ctr2])*params[ctr2];
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
			writer = new BufferedWriter(new FileWriter(new File("/Users/anilpa/Desktop/opdata/fakeOpData.csv"), false));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		SynthDataGenerator sgen = new SynthDataGenerator(writer, 4, 500);
		try {
			sgen.generate();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	
}
