#include <stdio.h>

int main(){

    // Declare a variable
    int a, b;

    // Get some integers.
    printf("Input your numbers (n m): ");
    scanf("%d %d", &a, &b);
    printf("\n\n %d x %d = %d", a, b, a*b);

    return 0;
}