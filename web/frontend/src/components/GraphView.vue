<template>
<div id="graphview">
    <h2 class="text-center" v-if="summaryWrapper">Sentiment Distribution</h2>
    <PieChart :options="{responsive: true, maintainAspectRatio: false}" :chart-data="summaryWrapper" id="main-chart" v-if="summaryWrapper"/>
    <b-row>
        <b-col cols="4" offset="4">
            <b-button block variant="info" v-on:click="fetchData">
                <icon name="refresh"></icon> Load Pie Chart
            </b-button>
        </b-col>
    </b-row>
</div>
</template>

<script>
import $ from 'jquery'
import 'vue-awesome/icons/refresh'
import PieChart from './PieChart'

export default {
    name: 'GraphView',
    components: {
        PieChart
    },
    data() {
        return {
            summary: null,
            summaryWrapper: null
        }
    },
    methods: {
        fillData: function() {
            if (!this.summary) {
                return;
            }

            this.summaryWrapper = {
                labels: ['Positive', 'Negative', 'Unsure/None'],
                datasets: [{
                        data: [this.summary.positive, this.summary.negative, this.summary.unsure],
                        backgroundColor: ['#81c784', '#e57373', '#9e9e9e'],
                        hoverBackgroundColor: ['#519657', '#af4448', '#707070']
                }]
            }
        },
        fetchData: function() {
            var component = this;
            $.getJSON('/api/summary', function(data) {
                component.summary = data;
                component.fillData();
            });
        }
    }
}
</script>

<style scoped>
#graphview {
    padding: 15px 0 5px 0;
}

#main-chart {
    margin-top: 10px;
    margin-bottom: 20px;
}
</style>
