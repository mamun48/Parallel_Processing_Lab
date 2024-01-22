#include <iostream>
#include <mpi.h>
using namespace std;
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time, end_time;
    int Num_matrices;
    int Mat_row;
    int Mat_col;
    int MatB_col;
    if (rank == 0)
    {
        printf("Enter Number of matrices\n");
        scanf("%d", &Num_matrices);
        printf("Enter Number of row of Mat A\n");
        scanf("%d", &Mat_row);
        printf("Enter Number of row of Mat B\n");
        scanf("%d", &Mat_col);
        printf("Enter Number of colum of Mat B\n");
        scanf("%d", &MatB_col);
    }

    // Announcing the value of four variable to each core using MPI_Bcast
    MPI_Bcast(&Num_matrices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Mat_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Mat_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MatB_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int root = 0;

    // Declare the Matrix A,B and result
    int matricesA[Num_matrices][Mat_row][Mat_col];
    int matricesB[Num_matrices][Mat_col][MatB_col];
    srand(time(0));

    // Take input using random function
    if (rank == root)
    {
        for (int i = 0; i < Num_matrices; i++)
        {
            for (int row = 0; row < Mat_row; row++)
            {
                for (int colum = 0; colum < Mat_col; colum++)
                {
                    matricesA[i][row][colum] = rand() % 10;
                }
            }

            for (int row = 0; row < Mat_col; row++)
            {
                for (int colum = 0; colum < MatB_col; colum++)
                {
                    matricesB[i][row][colum] = rand() % 10;
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Local MatrixA and MatrixB declaration and distribute each process using Scatter function
    int localMatrixA[Num_matrices / size][Mat_row][Mat_col];
    int localMatrixB[Num_matrices / size][Mat_col][MatB_col];
    int result[Num_matrices / size][Mat_row][MatB_col];

    MPI_Scatter(matricesA, (Num_matrices / size) * Mat_row * Mat_col, MPI_INT, localMatrixA, (Num_matrices / size) * Mat_row * Mat_col, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Scatter(matricesB, (Num_matrices / size) * Mat_col * MatB_col, MPI_INT, localMatrixB, (Num_matrices / size) * Mat_col * MatB_col, MPI_INT, root, MPI_COMM_WORLD);

    // Matrix multiplication on each core
    for (int i = 0; i < Num_matrices / size; i++)
    {
        for (int j = 0; j < Mat_row; j++)
        {
            for (int k = 0; k < MatB_col; k++)
            {
                result[i][j][k] = 0;
                for (int l = 0; l < Mat_col; l++)
                {
                    result[i][j][k] += localMatrixA[i][j][l] * localMatrixB[i][l][k];
                }
            }
        }
    }
    end_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d take time = %f second\n", rank, end_time - start_time);

    // Answeres are gathered
    int gatherMatrix[Num_matrices][Mat_row][MatB_col];
    MPI_Gather(result, (Num_matrices / size) * Mat_row * MatB_col, MPI_INT, gatherMatrix, (Num_matrices / size) * Mat_row * MatB_col, MPI_INT, root, MPI_COMM_WORLD);

    // Print the resultant matrices
    if (rank == root)
    {
        cout << "All " << Num_matrices << " are :" << endl;
        for (int i = 0; i < Num_matrices; i++)
        {
            cout << "Matrix " << i << ": " << endl;
            for (int j = 0; j < Mat_row; j++)
            {
                for (int k = 0; k < MatB_col; k++)
                {
                    cout << gatherMatrix[i][j][k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    MPI_Finalize();
    return 0;
}
