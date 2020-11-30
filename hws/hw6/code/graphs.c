#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int label;
    struct Node *next;
} node_t;

typedef struct Graph {
    int n;
    float p;
    node_t **nodes;
} graph_t;

void add_to_adj(node_t *source, node_t *dest) {
    node_t *ptr = source;
    while (ptr->next != NULL) { ptr = ptr->next; }
    ptr->next = dest;
}

graph_t make_graph(int n, float p) {
    graph_t *graph = NULL;
    graph = (graph_t *) malloc(sizeof(graph_t));
    graph->n = n;
    graph->p = p;

    // Allocate nodes
    for (int i = 0; i < n; i++) {
        graph->nodes[i] = (node_t *) malloc(sizeof(node_t));
    }

    // Connect vertices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && rand() < p) {
                add_to_adj(graph->nodes[i], graph->nodes[j]);
                add_to_adj(graph->nodes[j], graph->nodes[i]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    
}