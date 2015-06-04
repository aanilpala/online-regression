package regression.online.util;

public class CholeskyDecom {

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
	
	public static void main(String[] args) {
		
		// testing 
		
		//double[][] matrix = {{1.31,12.7,2.8}, {2.37,1.2,2.9}, {2.34,1.33,2.98}};
		double[][] matrix = {{4,12,-16}, {12,37,-43}, {-16,-43,98}};
		
		double[][] chk_decomposed = null;
		double[][] inv_chk_decom = null;
		double[][] gotta_be_identity = null;
		
		try {
			chk_decomposed = CholeskyDecom.chk_decom(matrix);
			inv_chk_decom = MatrixOp.invert_lower_tri_matrix(chk_decomposed);
			gotta_be_identity = MatrixOp.mult(matrix, MatrixOp.mult(MatrixOp.transpose(inv_chk_decom), inv_chk_decom));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		MatrixPrinter.print_matrix(matrix);
		MatrixPrinter.print_matrix(chk_decomposed);
		MatrixPrinter.print_matrix(inv_chk_decom);
		MatrixPrinter.print_matrix(gotta_be_identity);
		
		
		
	}
}
