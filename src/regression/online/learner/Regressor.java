package regression.online.learner;
import regression.online.model.Prediction;
import regression.online.util.NLInputMapper;


public abstract class Regressor {

	public int feature_count;
	NLInputMapper nlinmap;
	public boolean map2fs;
	private String name;
	
	public int id;
	
	int update_count;
	int burn_in_count;
	
	
	boolean verbouse = false;
	
	public Regressor(boolean map2fs2, int input_width) {
		
		this.map2fs = map2fs2;
		
		name = this.getClass().getName();
		
		if(map2fs) {
			nlinmap = new NLInputMapper(input_width, 2, true, true);
			this.feature_count = nlinmap.feature_dim;
		}
		else {
			this.feature_count = input_width;
		}
		
		update_count = 0;
		
	}
	
	public int getId() { return id;}
	
	public Prediction predict(double[][] dp) throws Exception { return null; };
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {}

	public int get_burn_in_number() {return burn_in_count; };

	public boolean is_tuning_time() {return false; };
	
	public String get_name() {
		return name.split("\\.")[3];  
	}
	
}
