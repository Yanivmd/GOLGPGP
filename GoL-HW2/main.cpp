/*
 Omer Katz, 3003457074, omerkatz@cs.technion.ac.il
 Yaniv David, 301226817, yanivd@tx.technion.ac.il
 */

#include "inc.h"


//using namespace std;



// make cells on margin of field dead
void clearMargin(byte* field, int sizeX, int sizeY);

// write field to stream (screen or file). Optionally, omit writing the margin.
void writeField(byte* field, int sizeX, int sizeY, int marginSize = 0,
		std::ostream& outs = std::cout);

// simple host implementation of computing the next step
void evolve(byte* orig, byte* next, int sizeX, int sizeY);

// print usage information
void printUsage(int argc, char** argv);

int writeBufToFile(std::string outfilename, std::string postfix, byte* buf, int fieldSizeX, int fieldSizeY)
{
	std::ofstream outf;
	outf.open((outfilename+postfix).c_str());     //< write to file
	if (outf)
	{
		// print to file without a 1-wide margin
		writeField(buf,fieldSizeX+2,fieldSizeY+2,1,outf);
		outf.close();
	}
	return 0;
}






int cpuSim(int iterations, byte* ptr1, byte* ptr2, int fieldSizeX, int fieldSizeY)
{
	for (int i=0; i<iterations; i++)
	{
		evolve(ptr1,ptr2,fieldSizeX+2,fieldSizeY+2);
		//writeField(ptr2,fieldSizeX+2,fieldSizeY+2);
		byte* tmp = ptr1; ptr1 = ptr2; ptr2 = tmp;
	}
	return 0;
}

int main(int argc, char** argv)
{
	const int NUM_ARGS = 3;

	if (argc != NUM_ARGS+1)
	{
	    printUsage(argc, argv);
	    return 1;
	}
	string infilename(argv[1]);
	string outfilename(argv[2]);
	int iterations = atoi(argv[3]);

	PatternBlock tblock;
	FieldReader reader;

	byte *in = new byte[(NUMBER_OF_ROWS+2)*(NUMBER_OF_COLS+2)];
#ifdef MEASUREMENTS
	byte *cpuout = new byte[(NUMBER_OF_ROWS+2)*(NUMBER_OF_COLS+2)];
#endif

	assert(reader.readFile(infilename));
	assert(reader.buildField(in,NUMBER_OF_COLS+2,NUMBER_OF_ROWS+2)); //< leave dead margin
	clearMargin(in,NUMBER_OF_COLS+2,NUMBER_OF_ROWS+2);

	//	memcpy(gpuin,cpuin,(fieldSizeY+2)*(fieldSizeX+2));
	writeBufToFile(outfilename,"in",in,NUMBER_OF_COLS,NUMBER_OF_ROWS);

	byte *gpuout = host(NUMBER_OF_COLS,NUMBER_OF_ROWS,in,iterations);
	writeBufToFile(outfilename,"gpu",gpuout,NUMBER_OF_COLS,NUMBER_OF_ROWS);

#ifdef MEASUREMENTS
	cpuSim(iterations,in,cpuout,NUMBER_OF_COLS,NUMBER_OF_ROWS);
	if (iterations % 2 == 0) {
		byte *tmp = cpuout;
		cpuout = in;
		in = tmp;
	}

	writeBufToFile(outfilename,"cpu",cpuout,NUMBER_OF_COLS,NUMBER_OF_ROWS);

	int errors = 0;
	for(int j=0;j<NUMBER_OF_ROWS;j++) {
		for(int i=0;i<NUMBER_OF_COLS;i++) {
			if (cpuout[(j+1)*(NUMBER_OF_COLS+2) + (i+1)] != gpuout[(j+1)*(NUMBER_OF_COLS+2) + (i+1)]) {
				errors +=1;
				std::cout << "fucked " << j << " " << i << " (CPU=[" << (int)(cpuout[(j+1)*(NUMBER_OF_COLS+2) + (i+1)]) << "],GPU=[" << (int)(gpuout[(j+1)*(NUMBER_OF_COLS+2) + (i+1)]) << "])\n";
				if (errors == 5)
					break;
			}
		}
		if (errors == 5)
			break;
	}

	if(errors > 0) {
		std::cout << "Errors Detected" << "\n";
	} else {
		std::cout << "all good\n";
	}

	delete[] cpuout;
#endif

	delete[] in;
	delete[] gpuout;

	return 0;
}

void clearMargin(byte* field, int sizeX, int sizeY)
{
	for (int r = 0; r < sizeY; r++)
	{
		field[r * sizeX] = DEAD;
		field[r * sizeX + sizeX-1] = DEAD;
	}
	std::fill_n(&field[0],sizeX,DEAD);
	std::fill_n(&field[sizeX*(sizeY-1)],sizeX,DEAD);
}



void writeField(byte* field, int sizeX, int sizeY, int marginSize,
		std::ostream& outs)
{
	// The field may contain a margin of dead cells that should not be printed.
	// If marginSize==0 then the entire field is printed.
	int m = marginSize;
	outs << (sizeX-2*m) << " " << (sizeY-2*m) << std::endl;
	byte* ptr = field;
	ptr += m * sizeX;   //< skip top margin
	for (int r = 0+m; r < sizeY-m; r++)
	{
		ptr += m;       //< skip left margin
		for (int c=0+m; c < sizeX-m; c++)
		{
			if (*ptr == DEAD)
				outs << '.';
			else
				outs << '*';
			ptr++;
		}
		ptr += m;       //< skip right margin
		outs << std::endl;
	}
}

void evolve(byte* orig, byte* next, int sizeX, int sizeY)
{
	std::fill_n(next,sizeX * sizeY, DEAD);
	for (int y = 1; y < sizeY-1; ++y)
	{
		for (int x = 1; x < sizeX-1; ++x)
		{
			byte* ptr = &orig[y * sizeX + x];
			int neighbors = 0;
			/*
            int state;
            if (ptr[0] == ALIVE)
                state = ALIVE;
            else
                state = DEAD;
			 */
			if (ptr[-1 * sizeX + -1] == ALIVE) neighbors++;
			if (ptr[-1 * sizeX +  0] == ALIVE) neighbors++;
			if (ptr[-1 * sizeX +  1] == ALIVE) neighbors++;
			if (ptr[ 0 * sizeX + -1] == ALIVE) neighbors++;
			if (ptr[ 0 * sizeX +  1] == ALIVE) neighbors++;
			if (ptr[ 1 * sizeX + -1] == ALIVE) neighbors++;
			if (ptr[ 1 * sizeX +  0] == ALIVE) neighbors++;
			if (ptr[ 1 * sizeX +  1] == ALIVE) neighbors++;

			if (neighbors == 3 ||
					(ptr[0] == ALIVE && neighbors == 2) )
				next[y * sizeX + x ] = ALIVE;
		}
	}
	clearMargin(next, sizeX, sizeY);
}

void printUsage(int argc, char** argv)
{
	cout << "Usage: " << argv[0] << " <input-file> <output-file> <num-iterations>" << endl;
}
