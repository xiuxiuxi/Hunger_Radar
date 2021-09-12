// enter code to define margin and dimensions for svg
const margin = { top: 0, right: 0, bottom: 80, left: 0 };
const width = 600 - margin.left - margin.right;
const height = 350 - margin.top - margin.bottom;

const margin_pie = { top: 0, right: 250, bottom: 0, left: 0 };
const width_pie = 500 - margin_pie.left - margin_pie.right;
const height_pie = 350 - margin_pie.top - margin_pie.bottom;

const margin_bar = { top: 0, right: 30, bottom: 50, left: 30 };
const width_bar = 500 - margin_bar.left - margin_bar.right;
const height_bar = 300 - margin_bar.top - margin_bar.bottom;

const margin_fake = { top: 0, right: 150, bottom: 0, left: 150 };
const width_fake = 550 - margin_fake.left - margin_fake.right;
const height_fake = 350 - margin_fake.top - margin_fake.bottom;

const zoom = d3.zoom()
    .scaleExtent([1, 8])
    .on("zoom", zoomed);

// enter code to create color scale
const colorScale = d3.scaleQuantile()
    .range(["#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"]);

// enter code to create color scale
const colorScaleScheme = d3.scaleOrdinal()
    .range(d3.schemeCategory10);

// enter code to create color scale
const colorScaleSchemeFake = d3.scaleOrdinal()
    .range([d3.schemeCategory10[2], d3.schemeCategory10[3]]);

// enter code to create svg
const svg = d3
    .select("#choropleth")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .attr("transform", `translate(${margin.left}, ${margin.top})`);

const map = svg
    .append("g")

// // Draw base
var svg2 = d3
    .select('#pieChart')
    .append('svg')
    .attr('width', width_pie + margin_pie.left + margin_pie.right)
    .attr('height', height_pie + margin_pie.top + margin_pie.bottom)

const pieChart = svg2
    .append("g")
    .attr("transform", "translate(" + width_pie / 2 + "," + height_pie / 2 + ")");

// create a tooltip
const tooltip = d3.select("body")
    .append("div")
    .style("opacity", 0)
    .attr("class", "tooltip")
    .style("background-color", "white")
    .style("border", "solid")
    .style("border-width", "2px")
    .style("border-radius", "5px")
    .style("padding", "5px");

// // Draw base
var svg3 = d3
    .select('#barChart')
    .append('svg')
    .attr('width', width_bar + margin_bar.left + margin_bar.right)
    .attr('height', height_bar + margin_bar.top + margin_bar.bottom);

const barChart = svg3
    .append("g")
    .attr('transform', `translate(${margin_bar.left}, ${margin_bar.top})`);

// // Draw base
var svg4 = d3
    .select('#table')
    .append('svg')
    .attr('width', width_bar + margin_bar.left + margin_bar.right)
    .attr('height', height_bar + margin_bar.top + margin_bar.bottom);

const table = svg4
    .append("g")
    .attr('transform', `translate(${margin_bar.left}, ${margin_bar.top})`);

// // Draw base
var svg5 = d3
    .select('#fakePieChart')
    .append('svg')
    .attr('width', width_fake +  margin_fake.left + margin_fake.right)
    .attr('height', height_fake + margin_fake.top + margin_fake.bottom);

const fakePieChart = svg5
    .append("g")
    .attr("transform", "translate(" + (width_pie * 0.5 + margin_fake.left) + "," + height_pie / 2 + ")");

const mapJson = "/static/topojson/ma_towns.json";
const cityCSV = "/static/csv/Business_MA_Res_City.csv";
const cityCategoryCSV = "/static/csv/cities_business.csv";
const reviewCSV = "/static/csv/business_polarity.csv";
const businessCSV = "/static/csv/Business_MA_Res.csv";
const fakeReivewsCSV = "/static/csv/fake_true.csv";

var cityColors = {};
var mapData = null;
var cityData = null;
var cityCategoryData = null;
var businessData = null;
var reviewData = null;
var fakeReivewData = null;

Promise.all([
    d3.json(mapJson),
    d3.dsv(",", cityCSV, function (d) {
        return {
            city: d.city.toLocaleUpperCase(),
            count: +d["count"],
        }
    }),
    d3.dsv(",", cityCategoryCSV, function (d) {
        return {
            city: d.city.toLocaleUpperCase(),
            category: d.categories,
            count: +d["count"]
        }
    }),
    d3.dsv(",", businessCSV, function (d) {
        return {
            business_id: d.business_id,
            name: d.name,
            city: d.city.toLocaleUpperCase(),
            category: d.categories,
            address: d.address,
            star: +d["stars"]
        }
    }),
    d3.dsv(",", reviewCSV, function (d) {
        return {
            business_id: d.business_id,
            review_score: +d["polarity"]
        }
    }),
    d3.dsv(",", fakeReivewsCSV, function (d) {
        return {
            city: d.city,
            total_true: +d["total_true"],
            total_fake: +d["total_fake"]
        }
    })
]).then(
    data => {
        mapData = data[0];
        cityData = data[1];
        cityCategoryData = data[2];
        businessData = data[3];
        reviewData = data[4];
        fakeReivewData = data[5];

        drawMap();
        drawCitiesTable();
    }
);

function drawMap() {
    cityData = cityData.filter(x => x.count > 2);

    var all_counts = cityData.map(d => d.count);
    var all_cities = cityData.map(d => d.city.toLocaleUpperCase());

    colorScale.domain(all_counts);

    map
        .selectAll("path")
        .data(topojson.feature(mapData, mapData.objects.towns).features)
        .enter()
        .append("path")
        .attr("d", d3.geoPath())
        .attr("class", "city")
        .attr("id", d => d.properties.TOWN)
        .on("click", mouseclicked)
        .on("mouseover", mouseOverChart)
        .on("mouseleave", mouseLeaveChart)
        .on("mousemove", mouseMoveChart)
        .style("stroke", "white")
        .style("fill", function (d) {
            if (all_cities.includes(d.properties.TOWN)) {
                return colorScale(cityData.find(({ city }) => city === d.properties.TOWN).count);
            } else {
                return "gainsboro";
            }
        });

    map.attr("transform", `translate(${-800}, ${-160}) scale(2.00)`);

    svg
        .append("g")
        .append("text")
        .text("Num of Restaurants")
        .attr("transform", `translate(${width - 160}, ${60})`)
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")

    // Draw Legend
    var legend = d3.legendColor()
        .labelFormat(d3.format(",.0f"))
        .scale(colorScale);

    svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 160}, ${70})`);

    svg.select(".legend")
        .call(legend);
}


function drawCitiesTable() {
    $("#table-cities").empty();

    var cData = prepareCityTableData();
    cData.sort((a, b) => a.count - b.count);
    cData.forEach((data) => {
        $("#table-cities").prepend(`<tr><td><a href="javascript:mouseclicked(${data.city})">${data.city}</a></td><td>${data.count}</td><td class='text-right'>${data.value.star.toFixed(2)}</td><td class='text-right'>${data.value.review_score.toFixed(2)}</td></tr>`);
    });
}

function drawPieChart(cityName) {
    pieChart.selectAll("*").remove();

    var pData = preparePieChartData(cityName);
    var categories = pData.map(x => x.key);

    pData = pData.map(x => x.values);
    var counts = pData.map(x => x[0].count);
    var total = d3.sum(counts);

    colorScaleScheme.domain(categories);

    var pie = d3.pie()
        .sort(null)
        .value(function (d) { return d.value[0].count; })

    var data_ready = pie(d3.entries(pData))

    var radius = Math.min(width_pie, height_pie) / 2

    var arc = d3.arc()
        .innerRadius(radius * 0.5)
        .outerRadius(radius * 0.8);

    // Build the pie chart
    pieChart
        .selectAll("allSlices")
        .data(data_ready)
        .enter()
        .append("path")
        .attr("d", arc)
        .on("mouseover", mouseOverChart)
        .on("mouseleave", mouseLeaveChart)
        .on("mousemove", mouseMoveChart)
        .attr("id", function (d) { return (d.data.value[0].category + ": " + (d.data.value[0].count / total * 100).toFixed(2).toString() + "%") })
        .attr('fill', function (d) { return (colorScaleScheme(d.data.value[0].category)) })
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 1);

    // Draw Legend
    var size = 20
    pieChart
        .selectAll("legendColor")
        .data(categories)
        .enter()
        .append("rect")
        .attr("x", 200)
        .attr("y", function (d, i) { return -120 + i * (size + 5) })
        .attr("width", size)
        .attr("height", size)
        .style("fill", function (d) { return colorScaleScheme(d) })

    // Add one dot in the legend for each name.
    pieChart
        .selectAll("legendLabels")
        .data(categories)
        .enter()
        .append("text")
        .attr("x", 200 + size * 1.2)
        .attr("y", function (d, i) { return -120 + i * (size + 5) + (size / 2) })
        .text(function (d) { return d })
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")

    // Add count
    pieChart
        .selectAll("count")
        .data(counts)
        .enter()
        .append("text")
        .attr("x", 320)
        .attr("y", function (d, i) { return -120 + i * (size + 5) + (size / 2) })
        .text(function (d) { return d })
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")
}

function drawBarChart(cityName) {
    barChart.selectAll("*").remove();

    var bData = prepareBarChartData(cityName);

    var yMax_bar = d3.max(bData.map(x => x.value));
    var yMin_bar = d3.min(bData.map(x => x.value));

    colorScaleScheme.domain(bData.map(x => x.key));

    // Scale data
    var xScale_bar = d3
        .scaleBand()
        .domain(bData.map(x => x.key))
        .range([0, width_bar])
        .padding(0.1);

    var yScale_bar = d3
        .scaleLinear()
        .domain([yMin_bar - 0.2, yMax_bar])
        .range([height_bar, 0]);

    // Draw x axis
    var xAxis = d3
        .axisBottom(xScale_bar)

    barChart
        .append("g")
        .attr("transform", "translate(0," + height_bar + ")")
        .call(xAxis)
        .selectAll("text")
        .attr("y", 0)
        .attr("x", 9)
        .attr("dy", ".35em")
        .attr("transform", "rotate(40)")
        .style("font-weight", "bold")
        .style("text-anchor", "start");

    // Draw y axis
    var yAxis = d3
        .axisLeft(yScale_bar)
        .tickSize(5)
        .tickSizeInner(-width_bar);

    barChart
        .append("g")
        .call(yAxis);

    var bars = barChart
        .selectAll(".bar")
        .data(bData)
        .enter()

    bars
        .append("rect")
        .attr("class", "bar")
        .attr("id", function (d) { return (d.key + " : " + d.value.toString()) })
        .on("mouseover", mouseOverChart)
        .on("mouseleave", mouseLeaveChart)
        .on("mousemove", mouseMoveChart)
        .attr("x", function (d) { return xScale_bar(d.key); })
        .attr("width", xScale_bar.bandwidth())
        .attr("y", function (d) { return yScale_bar(d.value); })
        .attr("height", function (d) { return height_bar - yScale_bar(d.value); })
        .attr("fill", function (d) { return colorScaleScheme(d.key); })
        .style("opacity", 1);

    bars.append("text")
        .attr("x", function (d) { return xScale_bar(d.key) + 12; })
        .attr("y",  function (d) { return yScale_bar(d.value) + 15;  }) 
        .text(function(d) { return d.value; });
}

function drawFakePieChart(cityName) {
    fakePieChart.selectAll("*").remove();

    var pData = prepareFakePieChartData(cityName);
    var total = d3.sum(Object.values(pData));

    colorScaleSchemeFake.domain(Object.keys(pData));

    var pie = d3.pie()
        .value(function (d) { return d.value; })

    var data_ready = pie(d3.entries(pData))

    var radius = Math.min(width_fake, height_fake) / 2

    var arc = d3.arc()
        .innerRadius(0)
        .outerRadius(radius);

    // Build the pie chart
    fakePieChart
        .selectAll("allSlices")
        .data(data_ready)
        .enter()
        .append("path")
        .attr("d", arc)
        .on("mouseover", mouseOverChart)
        .on("mouseleave", mouseLeaveChart)
        .on("mousemove", mouseMoveChart)
        .attr("id", function (d) { return (d.data.key + ": " + ( (d.data.value / total * 100).toFixed(2)).toString() + "%") })
        .attr('fill', function (d) { return (colorScaleSchemeFake(d.data.key)) })
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 1);
    
    fakePieChart
        .selectAll('allPolylines')
        .data(data_ready)
        .enter()
        .append('polyline')
        .attr("stroke", "black")
        .style("fill", "none")
        .attr("stroke-width", 1)
        .attr('points', function (d) {
            var posA = arc.centroid(d)
            var posB = arc.centroid(d) 
            var posC = arc.centroid(d);
            var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 
            posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1); 
            return [posA, posB, posC]
        })

    fakePieChart
        .selectAll('allLabels')
        .data(data_ready)
        .enter()
        .append('text')
        .text(function (d) { return d.data.key + ": " + d.data.value.toString() })
        .attr('transform', function (d) {
            var pos = arc.centroid(d);
            var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
            pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
            return 'translate(' + pos + ')';
        })
        .style('text-anchor', function (d) {
            var midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
            return (midangle < Math.PI ? 'start' : 'end')
        })
}


function drawTable(cityName) {
    $("#table-body").empty();

    var tData = prepareTableData(cityName);
    tData.sort((a, b) => a.review_score - b.review_score).slice(0, 5);
    tData.forEach((data) => {
        $("#table-body").prepend(`<tr><td><a href="https://www.google.ca/maps/search/${data.name} ${data.address} ${data.city}" target="_blank">${data.name}</a></td><td>${data.category}</td><td>${data.address}</td><td class='text-right'>${data.star}</td><td class='text-right'>${data.review_score.toFixed(2)}</td></tr>`);
    });
}

function zoomed(event) {
    map.attr("transform", event.transform)
}

function mouseOverChart(d) {
    d3.select(this)
        .style("stroke", "black")
        .style("opacity", "0.3");

    tooltip
        .style("opacity", 1);
}

function mouseMoveChart(d) {
    tooltip
        .html(d.target.id)
        .style("left", (d.pageX) + "px")
        .style("top", (d.pageY - 40) + "px");
}

function mouseLeaveChart(d) {
    d3.select(this)
        .style("stroke", "white")
        .style("opacity", "1");

    tooltip
        .style("opacity", 0);
}

function removeAll() {
    d3
        .select("#selected-city")
        .text("");

    d3
        .select("#city-color")
        .style("background-color", "white");

    barChart.selectAll("*").remove();
    pieChart.selectAll("*").remove();
    table.selectAll("*").remove();
    fakePieChart.selectAll("*").remove();

    $(".city-info").css({ opacity: 0 });
}

function mouseclicked(d) {
    if (typeof d.target != "undefined") {
        var cityColor = d.target.style.fill;
        var cityName = d.target.id;
    } else {
        var cityColor = d.style.fill;
        var cityName = d.id;
    }


    if (cityColor != "gainsboro") {

        d3
            .select("#selected-city")
            .text(cityName);

        d3
            .select("#city-color")
            .style("background-color", cityColor);

        drawPieChart(cityName);
        drawBarChart(cityName);
        drawTable(cityName);
        drawFakePieChart(cityName);

        $(".city-info").css({ opacity: 1 });
    } else {
        removeAll();
    }
}

function prepareTableData(cityName) {
    var reduced = businessData.filter(x => x.city == cityName);

    var merged = [];
    for (var i = 0; i < reduced.length; i++) {
        merged.push({
            ...reduced[i],
            ...(reviewData.find((itmInner) => itmInner.business_id === reduced[i].business_id))
        }
        );
    }

    var result = [];
    var i = 0;
    while (i < 5) {
        i += 1;
        var curr_max = d3.max(merged.map(x => x.review_score));
        var curr = merged.find(x => x.review_score == curr_max);
        result.push(curr);
        var index = merged.indexOf(curr);
        if (index > -1) {
            merged.splice(index, 1);
          }
    }

    return result;
}

function prepareCityTableData() {
    var merged_business = [];
    for (var i = 0; i < businessData.length; i++) {
        merged_business.push({
            ...businessData[i],
            ...(reviewData.find((itmInner) => itmInner.business_id === businessData[i].business_id))
        }
        );
    }

    var nested_data = d3.nest()
        .key(function (d) { return d.city; })
        .rollup(function (leaves) { return { "review_score": d3.mean(leaves.map(x => x.review_score)), "star": d3.mean(leaves.map(x => x.star)) } })
        .entries(merged_business);


    var merged = [];
    for (var i = 0; i < cityData.length; i++) {
        merged.push({
            ...cityData[i],
            ...(nested_data.find((itmInner) => itmInner.key === cityData[i].city))
        }
        );
    }
    return merged;
}

function prepareBarChartData(cityName) {
    var reduced = businessData.filter(x => x.city == cityName);
    var merged = [];
    for (var i = 0; i < reduced.length; i++) {
        merged.push({
            ...reduced[i],
            ...(reviewData.find((itmInner) => itmInner.business_id === reduced[i].business_id))
        }
        );
    }

    var nested_data = d3.nest()
        .key(function (d) { return d.category; }).sortKeys(d3.ascending)
        .rollup(function (leaves) { return d3.mean(leaves.map(x => x.review_score)).toFixed(2) })
        .entries(merged);

    return nested_data;
}

function preparePieChartData(cityName) {
    var reduced = cityCategoryData.filter(x => x.city == cityName);

    var nested_data = d3.nest()
        .key(function (d) { return d.category; }).sortKeys(d3.ascending)
        .entries(reduced);


    return nested_data;
}

function prepareFakePieChartData(cityName) {
        var fakeData = fakeReivewData.filter(x => x.city == cityName);
        var result = {}
        result["True Reviews"] = fakeData[0].total_true;
        result["Fake Reviews"] = fakeData[0].total_fake;

        return result;
}