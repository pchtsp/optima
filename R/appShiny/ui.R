#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(timevis)

dashboardPage(
    dashboardHeader(title = "OptiPlan Demo"),
    dashboardSidebar(
        sidebarMenu(
            menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
            menuItem("Widgets", tabName = "widgets", icon = icon("th"))
        )
    ),
    ## Body content
    dashboardBody(
        tabItems(
            # First tab content
            tabItem(tabName = "dashboard",
                    fluidRow(
                        box(plotOutput("plot1", height = 250)),
                        
                        box(
                            title = "Controls",
                            sliderInput("slider", "Number of observations:", 1, 100, 50)
                        )
                    )
            ),
            
            # Second tab content
            tabItem(tabName = "widgets",
                    h2("Planning"),
                    timevisOutput('timevis1', height = 500),
                    radioButtons(inputId='option_graph',
                                 label='Type de graph',
                                 choices=c('Cravatte'='tie', 'Evolution'='evolution')),
                    h3(textOutput('resource')),
                    plotlyOutput('plotly1')
                    
            )
        )
    )
)
