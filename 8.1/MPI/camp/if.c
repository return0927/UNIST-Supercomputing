#include<stdio.h>
#include<math.h>


int main() {

	int num;
	printf("type an integar:");
	scanf("%d", &num);
	if(num%2==0)
	{
	printf("%d is even\n", num);
	}
	else
	{
	printf("%d is odd\n", num);
	}
}
