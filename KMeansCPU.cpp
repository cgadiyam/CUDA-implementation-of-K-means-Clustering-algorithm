#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <float.h>   //FLT_MAX
#include "KMeans.h"

void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k)
{
	bool exit = false;
	int count;
	Vector2 Center;
	//iterates until no data point changes its cluster
	while(!exit)
	{
		count = 0;
		exit = true;
		//Assignment of each data point to a cluster
		for(int i=0;i<n;i++)
		{
			float Min_Dist = FLT_MAX;
			int nearest_cluster = 0;
			for(int j=0;j<k;j++)
			{
				if(data[i].p.distSq(clusters[j]) < Min_Dist)
				{
					Min_Dist = data[i].p.distSq(clusters[j]);
					nearest_cluster = j;
				}
			}
			if(nearest_cluster != data[i].cluster)
			{
				data[i].cluster = nearest_cluster;
				exit = false;
			}
		}

		//calculation of new center for all 3 clusters
		for(int i=0;i<k;i++)
		{
			count = 0;
			Center.x = 0;
			Center.y = 0;
			for(int j=0;j<n;j++)
			{
				if(data[j].cluster == i)
				{
					Center.x += data[j].p.x;
					Center.y += data[j].p.y;
					count++;
				}
			}
			if(count >0)
			{
				clusters[i].x = (Center.x)/count;
				clusters[i].y = (Center.y)/count;
			}
		}
	}
}