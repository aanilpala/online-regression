package regression.online.util;

import aima.core.util.math.*;

public class MatrixOp {

	public static double[][] mat_add(double m1[][], double m2[][]) throws Exception{
		
		if(m1[0].length != m2[0].length || m1[0].length != m2[0].length) throw new Exception("Not equal sized matrices");
		
		
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
	
	public static double[][] mat_subtract(double[][] m1, double[][] m2) throws Exception {
		
		if(m1[0].length != m2[0].length || m1[0].length != m2[0].length) throw new Exception("Not equal sized matrices");
		
		int num_col = m1[0].length;
		int num_row = m1.length;
		
		double[][] ans = new double[num_row][num_col]; 
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			for(int ctr2 = 0; ctr2 < num_col; ctr2++) {
				ans[ctr][ctr2] = m1[ctr][ctr2] - m2[ctr][ctr2]; 
			}
		}
		
		return ans;
	}
	
	public static double[][] identitiy_add(double m1[][], double coeff) throws Exception{
		
		if(m1[0].length != m1.length) throw new Exception("Not a square matrix");
		
		int num_col = m1[0].length;
		int num_row = m1.length;
		
		double[][] ans = new double[num_row][num_col]; 
		
		for(int ctr = 0; ctr < num_row; ctr++) {
			for(int ctr2 = 0; ctr2 < num_col; ctr2++) {
				if(ctr == ctr2) ans[ctr][ctr2] = m1[ctr][ctr2] + coeff;
				else ans[ctr][ctr2] = m1[ctr][ctr2];
			} 
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
	
	public static double[][] mult(double m1[][], double m2[][]) throws Exception{
		   if(m1.length == 0) return new double[0][0];
		   if(m1[0].length != m2.length) throw new Exception("Incompatible sized matrices for multiplication");
		 
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

	public static double trace(double[][] m) throws Exception {
		
		if(m.length != m[0].length) throw new Exception("Not a square matrix");
		
		double trace = 0;
		for(int ctr = 0; ctr < m.length; ctr++)
			trace += m[ctr][ctr];
		
		return trace;
	}
	
	public static double[][] chk_decom(double[][] m) throws Exception {
        
		if(m.length != m[0].length) throw new Exception("Not a square matrix");

		int size = m.length;
		
        double[] p = new double[size];
        double[][] result = new double[size][size];
        
        for(int ctr = 0; ctr < size; ctr++) {
        	for(int ctr2 = 0; ctr2 < size; ctr2++) {
        		result[ctr][ctr2] = m[ctr][ctr2];
        	}
        }
        
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                double sum = result[i][j];
                for (int k = i - 1; k >= 0; k--)
                    sum -= (result[i][k] * result[j][k]);
                if (i == j) {
                    if (sum < 0.0) throw new Exception("Matrix is not positive definite");
                    else p[i] = Math.sqrt(sum);
                }
                else result[j][i] = sum / p[i];
            }
        }
            
        for (int r = 0; r < size; r++) {
            result[r][r] = p[r];
            for (int c = r + 1; c < size; c++)
            	result[r][c] = 0;
        }
        return result;
    }
	
	public static double[][] fast_invert_psd(double[][] m) throws Exception {
		
		double[][] low_tri = chk_decom(m);
		double[][] low_tri_inv = invert_lower_tri_matrix(low_tri);
		
		return mult(transpose(low_tri_inv), low_tri_inv);
		
	}
	
	public static double get_det_of_psd_matrix(double[][] m) throws Exception {
		double[][] low_tri = chk_decom(m);
		double result = 1;
		
		for(int ctr = 0; ctr < m.length; ctr++)
			result *= low_tri[ctr][ctr];
		
		return Math.pow(result, 2);
	}
	
	public static double get_det_of_psd_matrix_2(double[][] m) throws Exception {
		
		LUDecomposition a = new LUDecomposition(new Matrix(m));
		
		return Math.pow(a.det(), 2);
	}
	
	public static double[][] invert_lower_tri_matrix(double[][] m) throws Exception {
		
		if(m.length != m[0].length) throw new Exception("Matrix is not square");
		int size = m.length;
		
		for(int ctr = 0; ctr < size; ctr++) {
			for(int ctr2 = 0; ctr2 > ctr; ctr2++) {
				if(m[ctr][ctr2] != 0) throw new Exception("Matrix is not lower triangular");
			}
		}
		
		double result[][] = new double[size][size];
		
		for(int ctr = 0; ctr < size; ctr++) {
			for(int ctr2 = 0; ctr2 < size; ctr2++) {
				double sum = 0;
				for(int ctr3 = ctr-1; ctr3 >= 0; ctr3--) {
					sum += m[ctr][ctr3]*result[ctr3][ctr2];
				}
				if(ctr2 != ctr) result[ctr][ctr2] = -1*sum/m[ctr][ctr];
				else result[ctr][ctr2] = (1-1*sum)/m[ctr][ctr];
			}
		}
		
		return result;
	}
	
	public static void main(String[] args) {
		
	}

	public static boolean isProportional(double[][] vec1, double[][] vec2) throws Exception {
		
		if(vec1.length != vec2.length || vec1[0].length != 1 || vec2[0].length != 1) throw new Exception("Size is not supported for this Op");
		
		if(vec1.length == 1) return false;
		
		double proportion = vec1[0][0]/vec2[0][0];
		
		for(int ctr = 1; ctr < vec1.length; ctr++) {
			double cur_proportion = vec1[ctr][0]/vec2[ctr][0];
			if(cur_proportion != proportion) return false;
		}
		
		return true;
	}
}
