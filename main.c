#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef struct matrix
{
    double **data;
    int rows;
    int cols;
} Matrix;

typedef struct layer {
    Matrix *weights; 
    Matrix *biases;
    int size;  
} Layer;


typedef struct network
{
    Layer **layers;
    int number_layers;

}Network;


void free_Matrix(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) 
    {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

double random_double(double min, double max)
{
    double scale = (double)rand() / (double)RAND_MAX;

    return min + scale * (max - min);
}
double sigmoid(double x)
{
   return  1.0 / (1.0 + exp(-x));
}
double sigmoid_prime(double x)
{
    return exp(-x) / pow(1+exp(-x), 2);
}
void matrix_sigmoid(Matrix *m)
{
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->data[i][j] = sigmoid(m->data[i][j]);
        }
    }
}


Matrix* create_Matrix(int rows, int cols)
{
    Matrix *m = malloc(sizeof(Matrix));
    
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * sizeof(double*));

    for(int i=0; i<rows; i++)
    {
        m->data[i]= calloc(cols, sizeof(double));
    }

    return m;
}
void init_Matrix(Matrix *m)
{
    for(int i=0; i<m->rows; i++)
    {
        for(int j=0; j<m->cols; j++)
        {
            m->data[i][j]=random_double(-1.0, 1.0);
        }
    }
}
void print_Matrix(Matrix *m)
{
    for(int i=0; i<m->rows; i++)
    {
        for(int j=0; j<m->cols; j++)
        {
            printf("%.2f  ", m->data[i][j]);
        }
        printf("\n");

    }
    printf("\n\n\n");
}
Matrix* multiply_Matrix(Matrix* A, Matrix* B)
{
    Matrix *C = create_Matrix(A->rows, B->cols);

    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < B->cols; j++)
        {
            double sum = 0;
            
            for(int k = 0; k < A->cols; k++) 
            {
                sum += A->data[i][k] * B->data[k][j];
            }       
            C->data[i][j] = sum;
        }
    }


    return C;
}
Matrix* add_Matrix(Matrix* A, Matrix* B)
{
    Matrix *C = create_Matrix(A->rows, A->cols);

    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < A->cols; j++)
        {   
            C->data[i][j] = A->data[i][j] + B->data[i][j];
        }
    }
    return C;
}
Matrix* sub_Matrix(Matrix* A, Matrix* B)
{
    Matrix *C = create_Matrix(A->rows, A->cols);

    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < A->cols; j++)
        {   
            C->data[i][j] = A->data[i][j] - B->data[i][j];
        }
    }
    return C;
}
Matrix* transpose_Matrix(Matrix* A)
{
    Matrix* C = create_Matrix(A->cols, A->rows);
    for(int i=0; i<A->cols; i++)
    {
        for(int j=0; j<A->rows; j++)
        {
            C->data[i][j]=A->data[j][i];
        }
    }
    return C;
}
Matrix* Hadamard_product(Matrix* A, Matrix* B)
{
    Matrix* C = create_Matrix(A->rows, A->cols);
    for(int i=0; i<A->rows; i++)
    {
        for(int j=0; j<A->cols; j++)
        {
            C->data[i][j] = A->data[i][j] * B->data[i][j];
        }
    }
    return C;
}
Matrix* scalar_multiply(Matrix* A, double scalar)
{
    Matrix* C = create_Matrix(A->rows, A->cols);
    for(int i=0; i<A->rows; i++)
    {
        for(int j=0; j<A->cols; j++)
        {
            C->data[i][j] = A->data[i][j] * scalar;
        }
    }
    return C;
}
int argmax(Matrix *m)
{
    int max_index = 0;
    double max_value = m->data[0][0];

    for(int i = 1; i < m->rows; i++)
    {
        if(m->data[i][0] > max_value)
        {
            max_value = m->data[i][0];
            max_index = i;
        }
    }
    return max_index;
}
Matrix* matrix_sigmoid_prime(Matrix* z) {
    Matrix* out = create_Matrix(z->rows, z->cols);
    for(int i=0; i<z->rows; i++) {
        for(int j=0; j<z->cols; j++) {
            double sig = sigmoid(z->data[i][j]);
            out->data[i][j] = sig * (1.0 - sig);
        }
    }
    return out;
}

Matrix** create_Batch(int number_layers)
{
   Matrix **batch = calloc(number_layers, sizeof(Matrix*));
   return batch;
}
Layer* create_Layer(int number_Neurons, int number_Weights)
{
    Layer *l = malloc(sizeof(Layer));
    l->weights = create_Matrix(number_Neurons, number_Weights);
    l->biases = create_Matrix(number_Neurons, 1);
    l->size= number_Neurons;
    init_Matrix(l->weights);

    return l;
}
Network* create_Network(int number_layers, int* number_neurons)
{
    Network* net = malloc(sizeof(Network));
    net->number_layers = number_layers;

    int number_connections = number_layers-1; 
    net->layers = malloc(number_connections * sizeof(Layer*));

    for(int i = 0; i<number_connections; i++)
    {
        net->layers[i] = create_Layer(number_neurons[i+1], number_neurons[i]);
    }
 
    return net;
}

Matrix* create_Input(double* input, int sizeof_input)
{
    Matrix *output = create_Matrix(sizeof_input, 1);
    for (int i=0; i<sizeof_input; i++)
    {
        output->data[i][0] = input[i];
    }
    return output;

}
Matrix* create_Output(int number)
{
    Matrix* Output = create_Matrix(10, 1);
    Output->data[number-1][0]=1;
    return Output;
}
Matrix* feed_Forward(Network *net, Matrix* input, Matrix **activations_batch, Matrix **z_batch)
{
    activations_batch[0] = input;
    
    Matrix *current_activation = input;

    for(int i = 0; i < net->number_layers - 1; i++)
    {
        Matrix *weights = net->layers[i]->weights;
        Matrix *biases = net->layers[i]->biases;
        
        Matrix *dot_prod = multiply_Matrix(weights, current_activation);

        Matrix *z_val = add_Matrix(biases, dot_prod);
        
        z_batch[i] = z_val; 
        
        free_Matrix(dot_prod);

        Matrix *a_val = create_Matrix(z_val->rows, z_val->cols);
        
        for(int r=0; r<z_val->rows; r++) {
            a_val->data[r][0] = z_val->data[r][0];
        }

        matrix_sigmoid(a_val);
        
        activations_batch[i+1] = a_val;
        
        current_activation = a_val;

    }

    return current_activation;
}

Matrix** delta_error(Network *net, Matrix* expected_value, Matrix **activations_batch, Matrix **z_batch)
{
    int number_connections = net->number_layers - 1;
    Matrix **delta = malloc(number_connections * sizeof(Matrix*));
    for(int i = number_connections - 1; i >= 0; i--)
    {
        Matrix *z = z_batch[i];
        Matrix *sp = matrix_sigmoid_prime(z); 

        if(i == number_connections - 1)
        {
            // δ = (a - y) ⊙ σ'(z)
            
            Matrix *a = activations_batch[i+1]; 
            
            // (a - y)
            Matrix *cost_derivative = sub_Matrix(a, expected_value);
            
            // (a - y) ⊙ σ'(z)
            delta[i] = Hadamard_product(cost_derivative, sp);
            
            free_Matrix(cost_derivative);
        }
        else
        {
            // δ = ((W_next)^T · δ_next) ⊙ σ'(z)
            
            Matrix *W_next = net->layers[i+1]->weights;
            Matrix *delta_next = delta[i+1];
            
            Matrix *Wt = transpose_Matrix(W_next);
            
            Matrix *propagated_error = multiply_Matrix(Wt, delta_next);
            
            delta[i] = Hadamard_product(propagated_error, sp);
            
            free_Matrix(Wt);
            free_Matrix(propagated_error);
        }

        free_Matrix(sp);
    }

    return delta;
}
Matrix** weight_Gradient(Network *net, Matrix **delta, Matrix **activations)
{
    int number_connections = net->number_layers - 1;
    Matrix **gradient = malloc(number_connections * sizeof(Matrix*));

    for(int i = 0; i < number_connections; i++)
    {
        Matrix *a_prev = activations[i];
        Matrix *a_prev_T = transpose_Matrix(a_prev);
        gradient[i] = multiply_Matrix(delta[i], a_prev_T);
        free_Matrix(a_prev_T);
    }

    return gradient;
}
Matrix** bias_Gradient(Network *net, Matrix **delta)
{
    int number_connections = net->number_layers - 1;
    Matrix **bias_grad = malloc(number_connections * sizeof(Matrix*));

    for(int i = 0; i < number_connections; i++)
    {        
        bias_grad[i] = create_Matrix(delta[i]->rows, delta[i]->cols);

        for(int r = 0; r < delta[i]->rows; r++)
        {
            bias_grad[i]->data[r][0] = delta[i]->data[r][0];
        }
    }

    return bias_grad;
}

void feed_Backwards(Network *net, Matrix **activations_batch, Matrix **z_batch, double learning_rate, Matrix *expected_value)
{
    
    Matrix **delta = delta_error(net, expected_value, activations_batch, z_batch);
    
    Matrix **bias_grad = bias_Gradient(net, delta);
    Matrix **weight_grad = weight_Gradient(net, delta, activations_batch);

    int number_connections = net->number_layers - 1;

    for(int i = 0; i < number_connections; i++)
    {
        
        Matrix* w_step = scalar_multiply(weight_grad[i], learning_rate);
        
        Matrix* new_weights = sub_Matrix(net->layers[i]->weights, w_step);
        
        free_Matrix(net->layers[i]->weights);
        net->layers[i]->weights = new_weights;
        
        free_Matrix(w_step);

        Matrix* b_step = scalar_multiply(bias_grad[i], learning_rate);
        Matrix* new_biases = sub_Matrix(net->layers[i]->biases, b_step);
        
        free_Matrix(net->layers[i]->biases);
        net->layers[i]->biases = new_biases;
        
        free_Matrix(b_step);
        free_Matrix(delta[i]);
        free_Matrix(weight_grad[i]);
        free_Matrix(bias_grad[i]);
    }

    free(delta);
    free(weight_grad);
    free(bias_grad);
}

void train(Network *net, double learning_rate, Matrix** expected_values, Matrix** input_values, int number_examples)
{

    Matrix** activations = create_Batch(net->number_layers);
    Matrix** z = create_Batch(net->number_layers - 1);

    for(int i = 0; i < number_examples; i++)
    {
        feed_Forward(net, input_values[i], activations, z);
        feed_Backwards(net, activations, z, learning_rate, expected_values[i]);
        for(int j = 1; j < net->number_layers; j++) 
        {
            free_Matrix(activations[j]);
        }
        for(int j = 0; j < net->number_layers - 1; j++) 
        {
            free_Matrix(z[j]);
        }

        //PROGRESS BAR
        if (i % 10 == 0 || i == number_examples - 1) 
        {
            float progress = (float)(i + 1) / number_examples;
            int barWidth = 50;
            
            printf("\r["); 
            int pos = barWidth * progress;
            for (int k = 0; k < barWidth; ++k) {
                if (k < pos) printf("=");
                else if (k == pos) printf(">");
                else printf(" ");
            }
            printf("] %d%% (%d/%d)", (int)(progress * 100.0), i + 1, number_examples);
            fflush(stdout); 
        }
    }    
    printf("\n"); 
    free(activations);
    free(z);
}
double Accuracy(Network *net, Matrix** expected_values, Matrix** input_values, int number_examples)
{
    Matrix** activations = create_Batch(net->number_layers);
    Matrix** z = create_Batch(net->number_layers - 1);
    
    int correct_guesses = 0;

    for(int i = 0; i < number_examples; i++)
    {
        Matrix* final_output = feed_Forward(net, input_values[i], activations, z);

        int predicted_index = argmax(final_output);
        
        int actual_index = argmax(expected_values[i]);

        if(predicted_index == actual_index)
        {
            correct_guesses++;
        }
        for(int j = 1; j < net->number_layers; j++) {
            free_Matrix(activations[j]);
        }
        for(int j = 0; j < net->number_layers - 1; j++) {
            free_Matrix(z[j]);
        }
    }

    free(activations);
    free(z);

    return (double)correct_guesses / (double)number_examples;
}


void load_mnist_csv(const char *filename, Matrix ***images, Matrix ***labels, int num_samples)
{
    FILE *fp = fopen(filename, "r");
    

    *images = malloc(num_samples * sizeof(Matrix*));
    *labels = malloc(num_samples * sizeof(Matrix*));

    char line[10000]; 
    
    fgets(line, 10000, fp); 

    printf("Loading %s...\n", filename);

    for (int i = 0; i < num_samples; i++)
    {
        if (fgets(line, 10000, fp) == NULL) break;

        (*labels)[i] = create_Matrix(10, 1);
        
        char *token = strtok(line, ",");
        int label_val = atoi(token);
        
        (*labels)[i]->data[label_val][0] = 1.0;

        (*images)[i] = create_Matrix(784, 1);
        
        for (int j = 0; j < 784; j++)
        {
            token = strtok(NULL, ",");
            double pixel = atof(token);
            
            (*images)[i]->data[j][0] = pixel / 255.0; 
        }

        if (i % 1000 == 0) printf("\rLoaded %d/%d", i, num_samples);
    }
    printf("\nDone loading.\n\n");
    fclose(fp);
}
void save_network(Network* net, const char* filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d\n", net->number_layers);
    fprintf(fp, "%d ", net->layers[0]->weights->cols); 

    for (int i = 0; i < net->number_layers - 1; i++) 
    {
        fprintf(fp, "%d ", net->layers[i]->weights->rows);
    }

    fprintf(fp, "\n");
    for (int i = 0; i < net->number_layers - 1; i++) 
    {
        Matrix *w = net->layers[i]->weights;
        Matrix *b = net->layers[i]->biases;

        for(int r = 0; r < w->rows; r++) 
        {
            for(int c = 0; c < w->cols; c++) 
            {
                fprintf(fp, "%lf\n", w->data[r][c]); 
            }
        }

        for(int r = 0; r < b->rows; r++) 
        {
            fprintf(fp, "%lf\n", b->data[r][0]);
        }
    }
    fclose(fp);
    printf("Network saved successfully to %s\n\n\n", filename);
}

Network* load_network(const char* filename)
{
    FILE *fp = fopen(filename, "r");
    

    int num_layers;
    fscanf(fp, "%d", &num_layers);

    int *number_neurons = malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++) {
        fscanf(fp, "%d", &number_neurons[i]);
    }

    Network *net = create_Network(num_layers, number_neurons);
    free(number_neurons); 

    for (int i = 0; i < net->number_layers - 1; i++) 
    {
        Matrix *w = net->layers[i]->weights;
        Matrix *b = net->layers[i]->biases;

        for(int r = 0; r < w->rows; r++) {
            for(int c = 0; c < w->cols; c++) {
                fscanf(fp, "%lf", &w->data[r][c]);
            }
        }

        for(int r = 0; r < b->rows; r++) {
            fscanf(fp, "%lf", &b->data[r][0]);
        }
    }

    fclose(fp);
    printf("\nNetwork loaded successfully from %s\n\n", filename);
    return net;
}

void print_ascii_art(Matrix *img)
{
    printf("\n\n\n--- INPUT VISUALIZATION ---\n");
    for(int i=0; i<28; i++)
    {
        for(int j=0; j<28; j++)
        {
            double val = img->data[i*28 + j][0];
            
            if(val > 0.1) printf("##"); 
            else printf("..");
        }
        printf("\n");
    }
    printf("---------------------------\n");
}


int main()
{
    srand(time(NULL));
    Network* net = NULL;
    const char* save_file = "network.txt";
    int option = 0;
    int save_choice = 0;

    printf("1 - Train Network \n");
    printf("2 - Load Network\n");
    printf("Pick your Option: ");
    scanf("%d", &option); 
    if(option == 1)
    {
        int number_neurons[] = {784, 128, 64, 10};
        
        net = create_Network(4, number_neurons);

        int num_train = 60000;
        Matrix **train_images;
        Matrix **train_labels;
        
        load_mnist_csv("mnist_train.csv", &train_images, &train_labels, num_train);

        printf("Starting Training \n\n");
        clock_t start = clock();
        
        train(net, 0.1, train_labels, train_images, num_train);
        
        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Training finished in %.2f seconds.\n\n", time_taken);
        
        double acc = Accuracy(net, train_labels, train_images, 1000); 
        printf("Training Accuracy: %.2f%%\n\n", acc * 100.0);
        
        printf("Save Network (0 - Yes / 1 - No) ? ");
        scanf("%d", &save_choice); 

        if(save_choice == 0) 
        {
            save_network(net, save_file);
        }

        for(int i=0; i<num_train; i++) 
        {
            free_Matrix(train_images[i]);
            free_Matrix(train_labels[i]);
        }

        free(train_images);
        free(train_labels);
    }
    else
    {
        net = load_network(save_file);
        if (net == NULL) return 1;
    }
    while(1)
    {
        printf("1 - Load 'my_drawing.csv' and Predict\n");
        printf("0 - Exit\n");
        printf("Pick your Option: ");
        scanf("%d", &option); 
        if(option == 1)
        {
            Matrix **user_image;
            Matrix **dummy_label;
        
        
            load_mnist_csv("my_drawing.csv", &user_image, &dummy_label, 1);
        
            printf("\nPredicting...\n");
            print_ascii_art(user_image[0]);

        
            Matrix** temp_act = create_Batch(net->number_layers);
            Matrix** temp_z = create_Batch(net->number_layers - 1);

            Matrix* final_output = feed_Forward(net, user_image[0], temp_act, temp_z);
        
            int prediction = argmax(final_output);
            printf("You drew a: %d\n\n", prediction);

        
            for(int i=1; i<net->number_layers; i++) free_Matrix(temp_act[i]); 
            for(int i=0; i<net->number_layers-1; i++) free_Matrix(temp_z[i]);
            free(temp_act);
            free(temp_z);

            free_Matrix(user_image[0]);
            free_Matrix(dummy_label[0]);
            free(user_image);
            free(dummy_label);

        }
        else {exit(0);}
    }

    return 0;
}