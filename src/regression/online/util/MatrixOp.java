package regression.online.util;

public class MatrixOp {

	public static double[][] mat_add(double m1[][], double m2[][]){
		
		if(m1[0].length != m2[0].length || m1[0].length != m2[0].length) {
			return null;
		}
		
		int num_col = m1[0].length;
		int num_row = m1.length;
		
		double[][] ans = new double[num_row][num_col]; 
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			for(int ctr2 = 0; ctr2 < num_col; ctr2++) {
				ans[ctr][ctr2] = m1[ctr][ctr2] + m2[ctr][ctr2]; 
			}
		}
		
		return ans;
		
	}
	
	public static double[][] identitiy_add(double m1[][], double coeff){
		
		if(m1[0].length != m1.length) {
			return null;
		}
		
		int num_col = m1[0].length;
		int num_row = m1.length;
		
		double[][] ans = new double[num_row][num_col]; 
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			ans[ctr][ctr] += coeff; 
		}
		
		return ans;
	}
	
	public static double[][] scalarmult(double[][] m, double coeff) {
		
		int num_col = m[0].length;
		int num_row = m.length;
		
		double[][] ans = new double[num_row][num_col]; 
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			for(int ctr2 = 0; ctr2 < num_col; ctr2++) {
				ans[ctr][ctr2] = coeff*m[ctr][ctr2];
			}
		}
		
		return ans;
	}

	public static double[][] transpose(double[][] m) {
		
		int num_col = m[0].length;
		int num_row = m.length;
		
		double[][] ans = new double[num_col][num_row];  
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			for(int ctr2 = 0; ctr2 < num_col; ctr2++) {
				ans[ctr2][ctr] = m[ctr][ctr2];
			}
		}
		
		return ans;
	}
	
	public static double[][] mult(double m1[][], double m2[][]){
		   if(m1.length == 0) return new double[0][0];
		   if(m1[0].length != m2.length) return null;
		 
		   int n = m1[0].length;
		   int m = m1.length;
		   int p = m2[0].length;
		 
		   double ans[][] = new double[m][p];
		 
		   for(int i = 0;i < m;i++){
		      for(int j = 0;j < p;j++){
		         for(int k = 0;k < n;k++){
		            ans[i][j] += m1[i][k] * m2[k][j];
		         }
		      }
		   }
		   return ans;
		}

	public static boolean isEqual(double[][] m1, double[][] m2) {
		
		if(m1.length != m2.length) return false;
		if(m1[0].length != m2[0].length) return false;
		
		int num_col = m1[0].length;
		int num_row = m1.length;
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			for(int ctr2 = 0; ctr2 < num_col; ctr2++) {
				if(m1[ctr][ctr2] != m2[ctr][ctr2]) return false; 
			}
		}
		
		return true;
	}
}
