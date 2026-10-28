/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include "hpcperf_roi.h"
#include "hpcperf_golden.h"
#include <stdio.h>
#include <ctype.h>     // isspace: the bounded file-list read in loadData()
#include <errno.h>     // [HPC-Performance-AI fix] strerror(errno) / ERANGE of the checked input handling below
#include <limits.h>    // INT_MIN / INT_MAX: the checked integer options
#include <math.h>      // isfinite: the checked -lat / -lng values
#include <stdlib.h>    // strtol / strtod
#include <string.h>    // strlen / strerror
#include <sys/time.h>
#include <float.h>
#include <vector>
#include <algorithm>   // std::sort of the captured top-k copy (tools/golden)
#include "cuda.h"

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors
// [HPC-Performance-AI fix] the longest path accepted: Linux PATH_MAX (4096 bytes including the NUL) - 1, i.e. the
// longest path open(2) accepts at all. Paths are read with this field width; a longer one is an error (exit 1).
#define NN_PATH_CHARS 4095
#define NN_STR_(x) #x
#define NN_STR(x) NN_STR_(x)
// [HPC-Performance-AI fix] upstream ignores every CUDA return code, so a failed allocation or copy went unnoticed
#define NN_CUDA_CHECK(call) do { cudaError_t nn_err_ = (call); if (nn_err_ != cudaSuccess) {                      \
    fprintf(stderr, "error: %s failed: %s\n", #call, cudaGetErrorString(nn_err_)); exit(1); } } while (0)


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
	int    i=0;
	float lat = 0, lng = 0;   // [HPC-Performance-AI fix] the defaults printUsage() documents; upstream left them uninitialized
	int quiet=0,timing=0,platform=0,device=0;

    std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[NN_PATH_CHARS + 1];   // [HPC-Performance-AI fix] was [100]: see parseCommandline()
	int resultsCount=10;

    // parse command line
    // [HPC-Performance-AI fix] 1 = -h (usage, exit 0 as upstream); 2 = an invalid command line, reported on stderr
    // (upstream: usage and exit 0, or a crash on a missing option value)
    int parsed = parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device);
    if (parsed) {
      printUsage();
      return parsed == 1 ? 0 : 1;
    }

    int numRecords = loadData(filename,records,locations);
    if (resultsCount > numRecords) resultsCount = numRecords;

    //for(i=0;i<numRecords;i++)
    //  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);


    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;


	// Scaling calculations - added by Sam Kauffman
	cudaDeviceProp deviceProp;
	NN_CUDA_CHECK(cudaGetDeviceProperties( &deviceProp, 0 ));
	cudaDeviceSynchronize();
	unsigned long maxGridX = deviceProp.maxGridSize[0];
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	NN_CUDA_CHECK(cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory ));
	cudaDeviceSynchronize();
	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
	dim3 gridDim( gridX, gridY );

	if ( DEBUG )
	{
		print( totalDeviceMemory ); // 804454400
		print( freeDeviceMemory );
		print( usableDeviceMemory );
		print( maxGridX ); // 65535
		print( deviceProp.maxThreadsPerBlock ); // 1024
		print( threadsPerBlock );
		print( maxThreads );
		print( blocks ); // 130933
		print( gridY );
		print( gridX );
	}

	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
	if (distances == NULL) {   // [HPC-Performance-AI fix] unchecked upstream
		fprintf(stderr, "error: cannot allocate the host distances of %d records\n", numRecords);
		exit(1);
	}
	NN_CUDA_CHECK(cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords));
	NN_CUDA_CHECK(cudaMalloc((void **) &d_distances,sizeof(float) * numRecords));

   /**
    * Transfer data from host to device
    */
    NN_CUDA_CHECK(cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice));

    /**
    * Execute kernel
    */
    HPCPERF_ROI_BEGIN_SYNC();  // tools/timing ROI: distance kernel, copy back, host k-nearest selection
    euclid<<< gridDim, threadsPerBlock >>>(d_locations,d_distances,numRecords,lat,lng);
    cudaDeviceSynchronize();

    //Copy data from device memory to host memory
    cudaMemcpy( distances, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );

	// find the resultsCount least distances
    findLowest(records,distances,numRecords,resultsCount);
    HPCPERF_ROI_END_SYNC();
    // [HPC-Performance-AI fix] after the ROI: an error of the kernel launch, the synchronization or the copy back
    // (all unchecked inside the ROI, as upstream) is still pending here and fails the run instead of printing garbage
    NN_CUDA_CHECK(cudaGetLastError());
    if (hpcperf_golden_active()) {
      // tools/golden: the checked outputs: the k reported distances (records[i].distance, printed below),
      // sorted ascending exactly as verify.py compares them (findLowest already leaves them ascending)
      std::vector<float> hpcperf_topk(resultsCount);
      for (int g = 0; g < resultsCount; g++) hpcperf_topk[g] = records[g].distance;
      std::sort(hpcperf_topk.begin(), hpcperf_topk.end());
      hpcperf_golden_output("topk_distances", hpcperf_topk.data(), resultsCount, "f32");
      // the identities of the k reported records (their 48-character db lines, as printed below) as a set: sorted
      // bytewise, so neither the report order nor the order of (near-)equal distances matters; and the distance
      // reported with each of them, in that order (the pairing record <-> distance)
      std::vector<int> hpcperf_ord(resultsCount);
      for (int g = 0; g < resultsCount; g++) hpcperf_ord[g] = g;
      std::sort(hpcperf_ord.begin(), hpcperf_ord.end(), [&](int a, int b) {
        return memcmp(records[a].recString, records[b].recString, 48) < 0; });
      std::vector<unsigned char> hpcperf_recs((size_t)resultsCount * 48);
      std::vector<float> hpcperf_rdist(resultsCount);
      for (int g = 0; g < resultsCount; g++) {
        memcpy(&hpcperf_recs[(size_t)g * 48], records[hpcperf_ord[g]].recString, 48);   // loadData: exactly 48 chars
        hpcperf_rdist[g] = records[hpcperf_ord[g]].distance;
      }
      hpcperf_golden_output("topk_records", hpcperf_recs.data(), hpcperf_recs.size(), "u8");
      hpcperf_golden_output("topk_record_distances", hpcperf_rdist.data(), resultsCount, "f32");
    }

    // print out results
    if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
    free(distances);
    //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);

}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[NN_PATH_CHARS + 1];   // [HPC-Performance-AI fix] was [64], read with an unbounded fscanf("%s"): the
	                                  // configured absolute .db paths (90 chars here) overflowed it (see README)
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
    if (flist == NULL) {   // [HPC-Performance-AI fix] unchecked upstream: feof(NULL) crashed
        fprintf(stderr, "error: cannot open the filelist %s: %s\n", filename, strerror(errno));
        exit(1);
    }
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		// [HPC-Performance-AI fix] bounded read (upstream: "%s\n" without a field width)
		if(fscanf(flist, "%" NN_STR(NN_PATH_CHARS) "s", dbname) != 1) {
            // [HPC-Performance-AI fix] upstream printed "error reading filelist" and exited 0. After a path the
            // whitespace is skipped up to the next path or EOF (below), so this read fails only for a list that
            // names no .db file at all (empty or blank) or on a read error (e.g. a directory)
            if (ferror(flist))
                fprintf(stderr, "error: cannot read the filelist %s: %s\n", filename, strerror(errno));
            else
                fprintf(stderr, "error: the filelist %s names no .db file\n", filename);
            exit(1);
        }
        int next = fgetc(flist);   // a path that fills the field and goes on does not fit: fail, never truncate
        if (next != EOF && !isspace(next)) {
            fprintf(stderr, "error: a .db path in the filelist %s is longer than %d characters\n", filename, NN_PATH_CHARS);
            exit(1);
        }
        while (next != EOF && isspace(next)) next = fgetc(flist);   // the rest of upstream's "\n" directive: skip
        if (next != EOF) ungetc(next, flist);                       // the whitespace (EOF after the last path)
        fp = fopen(dbname, "r");
        if(!fp) {
            // [HPC-Performance-AI fix] upstream: "error opening a db" on stdout (exit 1), without the name
            fprintf(stderr, "error: cannot open the .db file %s (listed in %s): %s\n", dbname, filename, strerror(errno));
            exit(1);
        }
        // read each record
        // [HPC-Performance-AI fix] upstream never checked a record: fgets(49) + one fgetc assume lines of exactly 48
        // characters + newline (the hurricane_gen format "%4d %2d %2d %2d %2d %-9s %5.1f %5.1f %4d %4d\n"); any other
        // line shifted every later record (garbage lat/lng), a last record without newline was dropped silently and a
        // read error (e.g. a directory) looped forever. Now such a file is an error (exit 1); valid files read as before.
        int line = 0;
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            if (fgets(record.recString,49,fp) == NULL) {   // end of the file (upstream: the same break below)
                if (ferror(fp)) {
                    fprintf(stderr, "error: cannot read the .db file %s: %s\n", dbname, strerror(errno));
                    exit(1);
                }
                break;
            }
            int nl = fgetc(fp); // newline
            line++;
            if (feof(fp)) {
                if (record.recString[strspn(record.recString, " \t\r\n")] != '\0') {
                    fprintf(stderr, "error: line %d of the .db file %s has no terminating newline\n", line, dbname);
                    exit(1);
                }
                break;   // only trailing whitespace after the last record
            }
            if (strlen(record.recString) != 48 || nl != '\n') {
                fprintf(stderr, "error: line %d of the .db file %s is not a record of 48 characters followed by a "
                                "newline\n", line, dbname);
                exit(1);
            }

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    if (recNum == 0) {   // [HPC-Performance-AI fix] upstream went on with 0 records: a zero-sized grid, SIGFPE
        fprintf(stderr, "error: the .db files listed in %s contain no records\n", filename);
        exit(1);
    }
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record tempRec;   // [HPC-Performance-AI fix] was `Record *tempRec` (see the swap below)
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    // [HPC-Performance-AI fix] upstream swapped through a pointer to records[i] taken before records[i] was
    // overwritten (`tempRec = &records[i]; records[i] = records[minLoc]; records[minLoc] = *tempRec;`), so
    // records[minLoc] got its own value back and the record that sat at position i was lost while its distance
    // moved to minLoc: once a lost record is among the topN it is reported as a duplicate of an earlier record
    // with the lost record's distance (cane4, -lat 30 -lng 90: correct up to -r 363; at -r 364 one duplicate, at
    // -r 42764 only 21447 distinct records). The distances were always right. Swap by value; the k <= 363
    // queries (the registered cane4-k5 included) print and capture byte-identical output.
    tempRec = records[i];
    records[i] = records[minLoc];
    records[minLoc] = tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

// [HPC-Performance-AI fix] option values are parsed completely: upstream's atoi / atof turned a missing value into
// a NULL dereference (SIGSEGV) and a malformed one ("abc", "5x", an out-of-range number) silently into 0 or a
// truncated value. strtol / strtod give the same value as atoi / atof for every well-formed number (atoi(s) is
// (int)strtol(s, NULL, 10), atof(s) is strtod(s, NULL)), so a valid command line runs exactly as before.
static const char *nn_option_value(int argc, char *argv[], int i) {
    if (i + 1 >= argc) {
      fprintf(stderr, "error: option %s needs a value\n", argv[i]);
      return NULL;
    }
    return argv[i + 1];
}

static int nn_parse_int(const char *opt, const char *s, int *out) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) {
      fprintf(stderr, "error: option %s needs an integer value, got '%s'\n", opt, s);
      return 1;
    }
    *out = (int)v;
    return 0;
}

static int nn_parse_float(const char *opt, const char *s, float *out) {
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE || !isfinite(v) || !isfinite((float)v)) {
      fprintf(stderr, "error: option %s needs a finite number, got '%s'\n", opt, s);
      return 1;
    }
    *out = v;   // the double -> float conversion of upstream's `*lat = atof(...)`
    return 0;
}

// returns 0 = run, 1 = help requested (-h / --help: usage, exit 0 as upstream), 2 = invalid command line
// (an error message on stderr; [HPC-Performance-AI fix] upstream printed the usage and exited 0, or crashed)
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d){
    int i;
    if (argc < 2) {
      fprintf(stderr, "error: the filelist (first argument) is missing\n");
      return 2;
    }
    if (strcmp(argv[1], "--help") == 0 ||                                         // upstream: --help crashed
        (argv[1][0] == '-' && strchr(argv[1], 'h') && strspn(argv[1] + 1, "hqt") == strlen(argv[1] + 1)))
      return 1;
    if (argv[1][0] == '-') {   // upstream took the option as the file name and ignored the real one
      fprintf(stderr, "error: the first argument must be the filelist, got the option %s\n", argv[1]);
      return 2;
    }
    // [HPC-Performance-AI fix] was strncpy(filename,argv[1],100), which leaves filename without a NUL terminator
    // for a path of >= 100 characters (e.g. an agent workspace's build directory); now a path that does not fit
    // is an error (exit 1), never truncated or read past the buffer
    if (strlen(argv[1]) > NN_PATH_CHARS) {
      fprintf(stderr, "error: the filelist path is longer than %d characters\n", NN_PATH_CHARS);
      exit(1);
    }
    strcpy(filename, argv[1]);   // fits: strlen(argv[1]) <= NN_PATH_CHARS, filename has NN_PATH_CHARS + 1 bytes

    // [HPC-Performance-AI fix] every argument after the filelist is one of the documented options, matched
    // exactly (upstream looked at the second character only, so "-long" was -lng, "-r5" took the NEXT argument
    // as the count, and an unknown option or a stray positional value -- e.g. "filelist 5 30 90" -- was ignored
    // silently, running another query with the defaults)
    for(i=2;i<argc;i++) {
      const char *a = argv[i], *v;
      if (strcmp(a, "-r") == 0) {             // number of results
        if (!(v = nn_option_value(argc, argv, i)) || nn_parse_int(a, v, r)) return 2;
        if (*r < 1) {                         // upstream: <= 0 printed nothing, exit 0
          fprintf(stderr, "error: option -r needs a positive number of results, got '%s'\n", v);
          return 2;
        }
        i++;
      } else if (strcmp(a, "-lat") == 0 || strcmp(a, "-lng") == 0) {
        if (!(v = nn_option_value(argc, argv, i)) || nn_parse_float(a, v, a[2] == 'a' ? lat : lng)) return 2;
        i++;
      } else if (strcmp(a, "-p") == 0 || strcmp(a, "-d") == 0) {   // platform / device (OpenCL leftovers, unused)
        if (!(v = nn_option_value(argc, argv, i)) || nn_parse_int(a, v, a[1] == 'p' ? p : d)) return 2;
        i++;
      } else if (strcmp(a, "--help") == 0) {
        return 1;
      } else if (a[0] == '-' && a[1] != '\0' && strspn(a + 1, "hqt") == strlen(a + 1)) {   // -h -q -t, combinable
        if (strchr(a, 'h')) return 1;
        if (strchr(a, 'q')) *q = 1;
        if (strchr(a, 't')) *t = 1;
      } else {
        fprintf(stderr, "error: unexpected argument '%s'\n", a);
        return 2;
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) { // both p and d must be specified if either are specified
      fprintf(stderr, "error: -p %d and -d %d must both be >= 0 or both < 0\n", *p, *d);
      return 2;
    }
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
